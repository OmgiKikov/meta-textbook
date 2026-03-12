"use client";

import { useEffect, useRef, useState } from 'react';
import { Transformer } from 'markmap-lib';
import { Markmap } from 'markmap-view';
import type { CSSProperties } from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import * as d3 from 'd3';

// Определяем интерфейс для узла
interface INode {
  v: string; // value (текст узла)
  p?: string; // путь в иерархии
  d?: number; // глубина (depth)
  children?: INode[]; // дочерние узлы
  content?: string; // контент узла (опционально)
  id?: string; // id узла
  [key: string]: any; // Другие свойства
}

interface MarkmapDisplayProps {
  markdown: string;
  className?: string;
  style?: CSSProperties;
  // Добавляем обратный вызов для запроса расширения узла
  onNodeExpand?: (nodeName: string) => void;
}

// Компонент модального окна для отображения информации о узле
interface NodeInfoModalProps {
  node: INode | null;
  onClose: () => void;
}

const NodeInfoModal = ({ node, onClose }: NodeInfoModalProps) => {
  if (!node) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" 
         onClick={onClose}>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-xl w-full max-h-[80vh] overflow-auto"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">{node.v}</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="prose dark:prose-invert max-w-none">
          <p className="text-sm text-muted-foreground">Глубина: {node.d ?? 0}</p>
          {node.children && node.children.length > 0 && (
            <div className="mt-4">
              <h3 className="text-base font-medium mb-2">Связанные понятия:</h3>
              <ul className="space-y-1">
                {node.children.map((child: INode, idx: number) => (
                  <li key={idx} className="py-1 px-2 bg-gray-100 dark:bg-gray-700 rounded-md text-sm">{child.v}</li>
                ))}
              </ul>
            </div>
          )}
          {node.content && (
            <div className="mt-6">
              <h3 className="text-lg font-medium mb-2">Содержание</h3>
              <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded-lg" 
                   dangerouslySetInnerHTML={{ __html: node.content }} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const transformer = new Transformer();

export function MarkmapDisplay({ markdown, className, style, onNodeExpand }: MarkmapDisplayProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const markmapRef = useRef<Markmap | null>(null);
  const [selectedNode, setSelectedNode] = useState<INode | null>(null);
  const [expandableNodes, setExpandableNodes] = useState<Set<string>>(new Set());
  const clickHandlerRef = useRef<((e: MouseEvent) => void) | null>(null);
  const transformRef = useRef<{ scale: number, x: number, y: number } | null>(null);
  const lastMarkdownRef = useRef<string>(markdown);
  const isFirstRender = useRef<boolean>(true); // Добавляем флаг первого рендера
  
  // Store onNodeExpand in a ref to avoid dependency changes
  const onNodeExpandRef = useRef(onNodeExpand);
  useEffect(() => {
    onNodeExpandRef.current = onNodeExpand;
  }, [onNodeExpand]);

  // Memoize the transformer result to avoid unnecessary regeneration
  const transformMarkdown = useRef<ReturnType<typeof transformer.transform> | null>(null);
  
  if (lastMarkdownRef.current !== markdown) {
    // Only transform markdown when it actually changes
    transformMarkdown.current = transformer.transform(markdown);
    lastMarkdownRef.current = markdown;
  }

  // Функция для определения, какие узлы можно расширить (листовые узлы второго уровня)
  const findExpandableNodes = (root: INode) => {
    const expandable = new Set<string>();
    
    const traverse = (node: INode, depth: number) => {
      // Узлы второго уровня (d=1) являются кандидатами на расширение
      if (depth === 1 || (node.d !== undefined && node.d === 1)) {
        expandable.add(node.v);
      }
      
      // Листовые узлы глубже третьего уровня тоже хорошие кандидаты
      if ((depth >= 2 || (node.d !== undefined && node.d >= 2)) && 
          (!node.children || node.children.length === 0)) {
        expandable.add(node.v);
      }
      
      if (node.children) {
        node.children.forEach(child => traverse(child, depth + 1));
      }
    };
    
    traverse(root, 0);
    return expandable;
  };

  // Добавляем стили для подсветки узлов при наведении
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.type = 'text/css';
    styleElement.innerHTML = `
      g.markmap-node {
        cursor: pointer;
      }
      
      g.markmap-node:hover > text {
        fill: hsl(var(--primary)) !important;
        font-weight: bold;
      }
      
      g.markmap-node:hover > path {
        stroke: hsl(var(--primary)) !important;
        stroke-width: 2px !important;
      }
      
      g.markmap-node.expandable > text {
        text-decoration: underline;
      }
      
      g.markmap-node.expandable::after {
        content: '+';
        font-size: 14px;
        fill: hsl(var(--primary));
        opacity: 0.8;
      }
      
      g.markmap-node.expandable:hover::after {
        opacity: 1;
      }
    `;
    document.head.appendChild(styleElement);

    return () => {
      if (document.head.contains(styleElement)) {
        document.head.removeChild(styleElement);
      }
    };
  }, []);

  // Create click handler once and store in ref
  useEffect(() => {
    // Настраиваем обработчик кликов
    const setupClickHandler = () => {
      // Skip if we already set up the handler
      if (clickHandlerRef.current) return;
      
      // Создаем новый обработчик клика
      const handleClick = (event: MouseEvent) => {
        const target = event.target as HTMLElement;
        let markmapNode = null;
        
        // Ищем ближайший узел markmap
        let currentElement: HTMLElement | null = target;
        while (currentElement && svgRef.current && !svgRef.current.isSameNode(currentElement)) {
          if (currentElement.tagName?.toLowerCase() === 'g' && 
              currentElement.classList?.contains('markmap-node')) {
            markmapNode = currentElement;
            break;
          }
          currentElement = currentElement.parentElement;
        }
        
        if (!markmapNode) return;
        
        // Находим текстовый элемент
        const textElement = markmapNode.querySelector('text') || 
                            markmapNode.querySelector('foreignObject div');
        
        if (!textElement) return;
        
        const nodeText = textElement.textContent?.trim() || '';
        console.log(`КЛИК ПО УЗЛУ: "${nodeText}"`);
        
        // По клику на любой узел вызываем onNodeExpand через ref
        if (onNodeExpandRef.current && nodeText) {
          onNodeExpandRef.current(nodeText);
          return;
        }
        
        // Ищем соответствующие данные узла для отображения модального окна
        if (markmapRef.current) {
          const nodes = (markmapRef.current as any)?.state?.nodes || [];
          
          // Поиск узла несколькими способами
          let node = nodes.find((n: any) => n.v === nodeText);
          
          if (!node) {
            const normalizedNodeText = nodeText.trim().replace(/\s+/g, ' ');
            node = nodes.find((n: any) => 
              n.v && n.v.trim().replace(/\s+/g, ' ') === normalizedNodeText);
          }
          
          if (!node) {
            const dataPath = markmapNode.getAttribute('data-path');
            if (dataPath) {
              node = nodes.find((n: any) => n.p === dataPath);
            }
          }
          
          if (!node) {
            node = nodes.find((n: any) => 
              n.v && nodeText && (n.v.includes(nodeText) || nodeText.includes(n.v)));
          }
          
          if (node) {
            setSelectedNode(node as INode);
          } else {
            // Создаем синтетический узел
            const syntheticNode: INode = {
              v: nodeText,
              d: parseInt(markmapNode.getAttribute('data-depth') || '0'),
              children: []
            };
            setSelectedNode(syntheticNode);
          }
        }
      };

      // Сохраняем обработчик в ref
      clickHandlerRef.current = handleClick;
      
      // Добавляем обработчик к SVG
      if (svgRef.current) {
        svgRef.current.addEventListener('click', handleClick);
      }
    };

    setupClickHandler();

    // Cleanup on unmount
    return () => {
      // Удаляем обработчик клика при размонтировании
      if (svgRef.current && clickHandlerRef.current) {
        svgRef.current.removeEventListener('click', clickHandlerRef.current);
        clickHandlerRef.current = null;
      }
    };
  }, []); // Only run once on mount

  // Setup markmap and update it when markdown changes
  useEffect(() => {
    if (!svgRef.current) return;

    // Initialize Markmap on first render or if it hasn't been created
    if (!markmapRef.current) {
      markmapRef.current = Markmap.create(svgRef.current, {
        duration: isFirstRender.current ? 500 : 0, // Анимации только при первом рендере
        embedGlobalCSS: true, // Embed basic Markmap CSS
        fitRatio: 0.95, // Fit ratio
        maxWidth: 300, // Max node width
        nodeMinHeight: 16, // Min node height
        paddingX: 8, // Node horizontal padding
        spacingHorizontal: 80, // Horizontal spacing
        spacingVertical: 5, // Vertical spacing
        autoFit: true, // Всегда устанавливаем autoFit=true для лучшего начального центрирования
        color: (node: any) => { // Function to determine branch color
          const depth = node.d ?? 0; // Node depth
          // Use theme colors based on depth
          const colors = [
            'hsl(var(--primary))',
            'hsl(var(--secondary))',
            'hsl(var(--accent))',
            'hsl(var(--muted-foreground))'
          ];
          return colors[depth % colors.length] || 'hsl(var(--foreground))';
        },
      });
      
      // Сохраняем текущую трансформацию при зуме/перемещении
      const svg = d3.select(svgRef.current);
      svg.on('zoom', () => {
        if (markmapRef.current) {
          const { scale, translate } = (markmapRef.current as any).state;
          transformRef.current = { 
            scale, 
            x: translate[0], 
            y: translate[1] 
          };
        }
      });
    }

    // Get markdown data from ref
    const { root, features } = transformMarkdown.current || transformer.transform(markdown);
    
    // Определяем узлы, которые можно расширить
    const expandable = findExpandableNodes(root as unknown as INode);
    setExpandableNodes(expandable);

    // Save current transform state before updating data
    const prevTransform = transformRef.current;

    // Update data in the existing Markmap instance with autoFit disabled for updates
    if (isFirstRender.current) {
      markmapRef.current?.setData(root);
      isFirstRender.current = false;
    } else {
      // Для обновлений отключаем автоматическую подгонку и анимации
      const mm = markmapRef.current as any;
      if (mm && mm.options) {
        // Временно отключаем анимации и автоподгонку
        const origAutoFit = mm.options.autoFit;
        const origDuration = mm.options.duration;
        mm.options.autoFit = false;
        mm.options.duration = 0; // Отключаем анимации при обновлении данных
        
        // Обновляем данные
        mm.setData(root);
        
        // Восстанавливаем оригинальные настройки
        mm.options.autoFit = origAutoFit;
        mm.options.duration = origDuration;
      } else {
        markmapRef.current?.setData(root);
      }
    }
    
    // Принудительно центрируем карту после обновления данных
    // Используем setTimeout чтобы дать карте время на рендеринг
    const centerMapTimeoutId = setTimeout(() => {
      if (markmapRef.current) {
        try {
          // Явно вызываем метод fit() для центрирования карты
          const mm = markmapRef.current as any;
          if (mm && typeof mm.fit === 'function') {
            mm.fit();
          }
        } catch (err) {
          console.error("Error centering mindmap:", err);
        }
      }
    }, 300); // Небольшая задержка для завершения рендеринга
    
    // Если у нас есть сохраненная трансформация, применяем её мгновенно, без задержек
    if (prevTransform) {
      const mm = markmapRef.current as any;
      if (mm && mm.state) {
        const { scale, x, y } = prevTransform;
        // Устанавливаем масштаб и позицию из сохраненного состояния немедленно
        mm.state.scale = scale;
        mm.state.translate = [x, y];
        mm.redraw();
      }
    }

    // Визуально отмечаем расширяемые узлы и добавляем символ ➕ к тексту
    // Уменьшим задержку для более быстрой отработки
    const markExpandableNodesTimeoutId = setTimeout(() => {
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        const nodes = svg.selectAll('g.markmap-node');
        
        nodes.each(function() {
          const node = d3.select(this);
          const text = node.select('text');
          if (text.size() > 0) {
            const nodeText = text.text().trim();
            if (expandableNodes.has(nodeText)) {
              node.classed('expandable', true);
              // Добавляем символ ➕, если еще не добавлен
              if (!text.text().endsWith('➕')) {
                text.text(nodeText + '  ➕');
              }
            }
          }
        });
      }
    }, 100); // Уменьшаем задержку с 500 до 100 мс

    // Cleanup specific to this effect
    return () => {
      clearTimeout(markExpandableNodesTimeoutId);
      clearTimeout(centerMapTimeoutId);
    };
  }, [markdown]); // Only depend on markdown

  // Add window resize handler to refit the mind map when window size changes
  useEffect(() => {
    const handleResize = () => {
      if (markmapRef.current) {
        const mm = markmapRef.current as any;
        if (mm && typeof mm.fit === 'function') {
          try {
            mm.fit();
          } catch (err) {
            console.error("Error fitting mindmap on resize:", err);
          }
        }
      }
    };
    
    // Debounce resize handler to avoid excessive updates
    let resizeTimeout: NodeJS.Timeout;
    const debouncedResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(handleResize, 200);
    };
    
    window.addEventListener('resize', debouncedResize);
    
    return () => {
      window.removeEventListener('resize', debouncedResize);
      clearTimeout(resizeTimeout);
    };
  }, []);

  // Force fit on markdown change
  useEffect(() => {
    // Reset transform state when markdown changes completely
    // This is important when switching to a completely new mindmap
    if (markdown !== lastMarkdownRef.current && markdown.trim() !== '') {
      transformRef.current = null;
    }
    
    // Small delay to let render complete then fit
    const fitTimeoutId = setTimeout(() => {
      if (markmapRef.current) {
        try {
          const mm = markmapRef.current as any;
          if (mm && typeof mm.fit === 'function') {
            console.log('Forcing fit on markdown change');
            mm.fit();
          }
        } catch (err) {
          console.error("Error fitting mindmap on markdown change:", err);
        }
      }
    }, 350);
    
    return () => clearTimeout(fitTimeoutId);
  }, [markdown]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (svgRef.current && clickHandlerRef.current) {
        svgRef.current.removeEventListener('click', clickHandlerRef.current);
      }
      markmapRef.current?.destroy();
      markmapRef.current = null;
    };
  }, []); // Only run on unmount

  return (
    <>
      <svg
        ref={svgRef}
        className={className}
        style={{ width: '100%', height: '100%', ...style }}
      />
      
      {/* Модальное окно с информацией о выбранном узле */}
      {selectedNode && (
        <NodeInfoModal 
          node={selectedNode} 
          onClose={() => setSelectedNode(null)} 
        />
      )}
    </>
  );
} 