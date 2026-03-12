"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import * as d3 from "d3"
import { cn } from "@/lib/utils"
import { NodePreview } from "@/components/node-preview"

interface Node {
  id: string
  label: string
  group: number
  val?: number
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
  type?: string
  originalData?: any
}

interface Link {
  source: string | Node
  target: string | Node
  value?: number
  relation_type?: string
  originalData?: any
}

interface GraphData {
  nodes: Node[]
  links: Link[]
}

interface D3GraphProps {
  data: GraphData | null
  width?: number
  height?: number
  className?: string
  onNodeClick?: (node: Node) => void
}

export function D3Graph({ data, width = 800, height = 600, className, onNodeClick }: D3GraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [previewNode, setPreviewNode] = useState<Node | null>(null)
  const [previewPosition, setPreviewPosition] = useState({ x: 0, y: 0 })
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [hoveredLink, setHoveredLink] = useState<Link | null>(null)
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 })
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMouseOverPreviewRef = useRef(false)

  // --- Новое: fit-to-screen только при первом рендере для новых данных ---
  const [hasFitToScreen, setHasFitToScreen] = useState(false);
  const prevDataRef = useRef<GraphData | null>(null);

  useEffect(() => {
    // Если data поменялась (новый граф) — сбрасываем флаг и фокус
    if (data !== prevDataRef.current) {
      setHasFitToScreen(false);
      prevDataRef.current = data;
      setFocusedNodeId(null);
      setPreviewNode(null);
    }
  }, [data]);

  // Словарь переводов типов связей
  const relationTypeTranslations: Record<string, string> = {
    "is_a": "подтип",
    "part_of": "компонент",
    "explains": "объяснение",
    "example_of": "пример",
    "contrasts_with": "противопоставление",
    "leads_to": "следствие",
    "analogous_to": "аналогия"
  };

  // Функция для получения человекочитаемого названия типа связи
  const getRelationTypeLabel = (relationType: string | undefined): string => {
    if (!relationType) return 'Связь';
    
    // Преобразуем к нижнему регистру для регистронезависимого поиска
    const normalizedType = relationType.toLowerCase();
    
    // Ищем в словаре перевод
    for (const [key, value] of Object.entries(relationTypeTranslations)) {
      if (normalizedType === key.toLowerCase()) {
        return value;
      }
    }
    
    // Если перевод не найден, возвращаем исходное значение с большой буквы
    return relationType.charAt(0).toUpperCase() + relationType.slice(1).replace(/_/g, ' ');
  };

  // Добавим ссылки для сохранения состояния
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const simulationRef = useRef<d3.Simulation<Node, any> | null>(null);

  // Debounced function to show node preview
  const showNodePreview = useCallback((node: Node, x: number, y: number) => {
    // Clear any existing timeouts
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }

    // Set position immediately to avoid jumps
    setHoverPosition({ x, y })

    // Set a timeout to show the preview
    previewTimeoutRef.current = setTimeout(() => {
      setPreviewNode(node)
    }, 600) // Increased delay for more stability
  }, [])

  // Handle mouse entering preview
  const handleMouseEnterPreview = useCallback(() => {
    isMouseOverPreviewRef.current = true
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }
  }, [])

  // Handle mouse leaving preview
  const handleMouseLeavePreview = useCallback(() => {
    isMouseOverPreviewRef.current = false
  }, [])

  // Debounced function to show link preview
  const showLinkPreview = useCallback((link: Link, x: number, y: number) => {
    // Clear any existing timeouts
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }

    // Set position immediately to avoid jumps
    setHoverPosition({ x, y })

    // Set a timeout to show the preview
    previewTimeoutRef.current = setTimeout(() => {
      setHoveredLink(link)
    }, 300) // Use a slightly shorter delay for links
  }, [])

  // Function to hide link preview with delay
  const hideLinkPreview = useCallback(() => {
    // Only set hide timeout if mouse is not over preview
    if (!isMouseOverPreviewRef.current) {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }

      hideTimeoutRef.current = setTimeout(() => {
        if (!isMouseOverPreviewRef.current) {
          setHoveredLink(null)
        }
      }, 200) // Shorter delay before hiding
    }
  }, [])

  // Clear all timeouts on unmount
  useEffect(() => {
    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current)
      }
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!svgRef.current) return;
    if (!data) return;
    if (!data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) return;
    if (!data.links || !Array.isArray(data.links)) return;
    
    console.log("Rendering graph with data:", data);

    // Сохраняем текущую трансформацию, если она есть
    const currentTransform = transformRef.current;
    
    // Clear any existing SVG content
    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3.select(svgRef.current)
    
    // Создаем container и сразу применяем сохраненную трансформацию
    const container = svg.append("g")
      .attr("transform", currentTransform ? `translate(${currentTransform.x}, ${currentTransform.y}) scale(${currentTransform.k})` : null);

    // Функция для определения цвета узла на основе его типа
    const getNodeColor = (node: Node) => {
      // Извлекаем тип узла из разных источников данных
      const nodeType = node.type || 
                     (node.originalData?.type) || 
                     (node.id?.includes('concept') ? 'concept' : 
                      node.id?.includes('person') ? 'person' : 
                      node.id?.includes('event') ? 'event' : 
                      node.id?.includes('object') ? 'object' : 
                      node.id?.includes('process') ? 'process' : 
                      node.id?.includes('location') ? 'location' : 
                      node.id?.includes('organization') ? 'organization' : 
                      node.id?.includes('term') ? 'term' : 
                      node.id?.includes('place') ? 'place' : 
                      node.id?.includes('work') ? 'work' : 
                      node.id?.includes('theory') ? 'theory' : 'unknown');
      
      // Определяем цвет на основе типа узла
      switch(nodeType.toLowerCase()) {
        case 'concept': return "#22c55e"; // Зеленый (как было ранее)
        case 'person': return "#ef4444"; // Красный
        case 'event': return "#3b82f6"; // Синий
        case 'location': 
        case 'place': return "#06b6d4"; // Голубой
        case 'organization': return "#f59e0b"; // Оранжевый
        case 'term': return "#8b5cf6"; // Фиолетовый
        case 'process': return "#ec4899"; // Розовый
        case 'object': return "#14b8a6"; // Бирюзовый
        case 'work': return "#f97316"; // Ярко-оранжевый
        case 'theory': return "#6366f1"; // Индиго
        default: return "#22c55e"; // По умолчанию зеленый
      }
    }

    // Define color scale for node groups
    const color = d3
      .scaleOrdinal<number, string>()
      .domain([0, 1, 2, 3, 4, 5])
      .range(["#22c55e", "#ef4444", "#3b82f6", "#06b6d4", "#f59e0b", "#8b5cf6"])

    // Добавляем маркер-стрелку для линий
    svg.append("defs").selectAll("marker")
      .data(["end"])
      .enter().append("marker")
      .attr("id", d => d)
      .attr("viewBox", "0 -5 12 10") // Увеличиваем viewBox для вытянутой стрелки
      .attr("refX", 17) // Увеличиваем расстояние от конца линии до маркера с 15 до 22
      .attr("refY", 0)
      .attr("markerWidth", 5) // Немного увеличиваем ширину для вытянутой стрелки
      .attr("markerHeight", 5) // Уменьшаем высоту маркера с 6 до 4
      .attr("orient", "auto")
      .append("path")
      .attr("fill", "#999")
      .attr("d", "M0,-2.5L8,0L0,2.5"); // Более вытянутая стрелка (длиннее и уже)

    // Process links to ensure they reference node objects
    const nodeMap = new Map<string, Node>()
    data.nodes.forEach((node) => nodeMap.set(node.id, node))

    console.log("Processing links:", data.links);
    console.log("NodeMap keys:", Array.from(nodeMap.keys()));

    // Функция для создания кривых путей между узлами
    const generateLinkPath = (d: any) => {
      const source = d.source as Node;
      const target = d.target as Node;
      
      // Координаты начала и конца
      const x1 = source.x!;
      const y1 = source.y!;
      const x2 = target.x!;
      const y2 = target.y!;
      
      // Определяем, должно ли ребро быть прямым
      // Проверяем явное указание прямого ребра или используем эвристику
      const isStraight = 
        d.isStraight || 
        (d.originalData?.isStraight) || 
        // Только рёбра с очень высоким значением strength делаем прямыми
        (d.strength >= 5) ||
        // Или используем детерминированную генерацию, чтобы примерно 15-20% рёбер были прямыми
        ((source.id.charCodeAt(0) + target.id.charCodeAt(0)) % 8 === 0);
      
      // Если ребро должно быть прямым - возвращаем прямую линию
      if (isStraight) {
        return `M${x1},${y1} L${x2},${y2}`;
      }
      
      // Для остальных рёбер создаем кривую
      // Рассчитываем среднюю точку между узлами
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;
      
      // Определяем смещение для кривизны
      const dx = x2 - x1;
      const dy = y2 - y1;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Чем ближе узлы, тем меньше изгиб
      // Но гарантируем минимальную кривизну для визуальной различимости
      const curvature = Math.min(0.3, Math.max(0.1, 40 / dist));
      
      // Рассчитываем смещение для контрольной точки
      // Перпендикулярно к линии, соединяющей начало и конец
      const offsetX = midX - dy * curvature;
      const offsetY = midY + dx * curvature;
      
      // Строим путь с квадратичной кривой Безье
      return `M${x1},${y1} Q${offsetX},${offsetY} ${x2},${y2}`;
    };

    // Подготовка связей, очень тщательная проверка и преобразование
    // Пытаемся обработать связи независимо от их формата
    const links = data.links
      .filter(link => {
        // Получаем ID источника
        const sourceId = typeof link.source === 'string' 
          ? link.source 
          : (link.source as any)?.id || (link.source as any)?.source_id;
        
        // Получаем ID цели
        const targetId = typeof link.target === 'string' 
          ? link.target 
          : (link.target as any)?.id || (link.target as any)?.target_id;
        
        console.log(`Checking link: ${sourceId} -> ${targetId}`);
        
        // Проверяем наличие узлов в nodeMap
        const hasSource = Boolean(sourceId && nodeMap.has(sourceId));
        const hasTarget = Boolean(targetId && nodeMap.has(targetId));
        
        if (!hasSource) console.warn(`Source node ${sourceId} not found`);
        if (!hasTarget) console.warn(`Target node ${targetId} not found`);
        
        return hasSource && hasTarget;
      })
      .map((link) => {
        // Снова получаем ID, так как мы отфильтровали некорректные связи
        const sourceId = typeof link.source === 'string' 
          ? link.source 
          : (link.source as any)?.id || (link.source as any)?.source_id;
          
        const targetId = typeof link.target === 'string' 
          ? link.target 
          : (link.target as any)?.id || (link.target as any)?.target_id;
        
        return {
          source: nodeMap.get(sourceId)!,
          target: nodeMap.get(targetId)!,
          value: link.value || 1,
          relation_type: link.relation_type || (link as any).originalData?.relation_type || 'связан с',
          originalData: link.originalData || link,
          strength: (link as any).strength || (link as any).originalData?.strength || 1
        };
      });
    
    console.log("Processed links:", links);

    // --- Подсветка: вычисляем разрешённые id ---
    let allowedNodeIds = new Set<string>();
    let allowedLinkIdx = new Set<number>();
    if (focusedNodeId) {
      allowedNodeIds.add(focusedNodeId);
      data.links.forEach((link, idx) => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any)?.id;
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any)?.id;
        if (sourceId === focusedNodeId) {
          allowedNodeIds.add(targetId);
          allowedLinkIdx.add(idx);
        }
        if (targetId === focusedNodeId) {
          allowedNodeIds.add(sourceId);
          allowedLinkIdx.add(idx);
        }
      });
    }

    // Create a force simulation
    const simulation = d3
      .forceSimulation<Node, Link>()
      .force(
        "link",
        d3
          .forceLink<Node, Link>(links)  // Явно передаем links в forceLink
          .id((d: Node) => d.id)
          .distance(180),  // Увеличиваем базовое расстояние между связанными узлами с 180 до 250
      )
      .force("charge", d3.forceManyBody().strength(-800)) // Значительно усиливаем отталкивание с -350 до -800
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(0.03)) // Уменьшаем силу притяжения к центру
      .force("y", d3.forceY(height / 2).strength(0.03)) // Уменьшаем силу притяжения к центру
      .force("collision", d3.forceCollide().radius((d: any) => {
        // Получаем значимость из разных источников данных
        const importance = d.originalData?.importance || d.val || 1;
        // Увеличиваем радиус коллизии
        return 8 + importance * importance * 1.2;
      })); // Увеличиваем радиус коллизии для предотвращения наложения

    // Create links - изменен порядок: сначала создаем связи, потом узлы
    const link = container
      .append("g")
      .attr("class", "links")
      .selectAll("path")
      .data(links)
      .enter()
      .append("path")
      .attr("fill", "none")
      .attr("stroke", (d, i) => {
        if (focusedNodeId) {
          return allowedLinkIdx.has(i) ? "#999" : "#bbb";
        }
        return "#999";
      })
      .attr("stroke-opacity", (d, i) => {
        if (focusedNodeId) {
          return allowedLinkIdx.has(i) ? 0.7 : 0.18;
        }
        return 0.6;
      })
      .attr("stroke-width", (d: any) => {
        const strength = d.strength || 1;
        return Math.max(3, Math.sqrt(strength) * 2.5);
      })
      .attr("marker-end", "url(#end)")
      .on("click", (event: any, d: Link) => {
        const [x, y] = d3.pointer(event, svg.node());
        setHoveredLink(d);
        setHoverPosition({ x, y });
        event.stopPropagation();
      })
      .style("cursor", "pointer");

    // Добавляем невидимую, но более широкую линию для лучшего взаимодействия (только клик)
    container
      .append("g")
      .attr("class", "link-hit-areas")
      .selectAll("path")
      .data(links)
      .enter()
      .append("path")
      .attr("fill", "none")
      .attr("stroke", "transparent")
      .attr("stroke-width", 12)
      .style("pointer-events", "stroke")
      .style("cursor", "pointer")
      .on("click", (event: any, d: Link) => {
        const [x, y] = d3.pointer(event, svg.node());
        setHoveredLink(d);
        setHoverPosition({ x, y });
        event.stopPropagation();
      });
    
    console.log("Number of link elements created:", link.size());

    // Create nodes and node groups
    const node = container
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(data.nodes)
      .enter()
      .append("g")
      .call(d3.drag<SVGGElement, Node>().on("start", dragstarted).on("drag", dragged).on("end", dragended))

    // Add invisible larger circle for better click detection
    node
      .append("circle")
      .attr("r", (d: Node) => {
        const importance = d.originalData?.importance || d.val || 1;
        return 8 + importance * importance * 0.8;
      })
      .attr("fill", "transparent")
      .style("pointer-events", "all")
      .on("click", (event: any, d: Node) => {
        const [x, y] = d3.pointer(event, svg.node());
        setPreviewNode(d);
        setPreviewPosition({ x, y });
        setFocusedNodeId(d.id);
        event.stopPropagation();
      });

    // Add visible circles to nodes
    node
      .append("circle")
      .attr("r", (d: Node) => {
        const importance = d.originalData?.importance || d.val || 1;
        return 4 + importance * importance * 0.8;
      })
      .attr("fill", (d: Node) => {
        if (focusedNodeId) {
          return allowedNodeIds.has(d.id) ? getNodeColor(d) : '#e5e7eb';
        }
        return getNodeColor(d);
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("pointer-events", "none")
      .attr("opacity", (d: Node) => {
        if (focusedNodeId) {
          return allowedNodeIds.has(d.id) ? 1 : 0.25;
        }
        return 1;
      });

    // Create a separate layer for labels that will be rendered last (on top of everything)
    const labels = container
      .append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(data.nodes)
      .enter()
      .append("text")
      .text((d: Node) => d.label)
      .attr("x", (d: Node) => d.x || 0)
      .attr("y", (d: Node) => {
        const importance = d.originalData?.importance || d.val || 1;
        return (d.y || 0) + 15 + importance * importance * 0.8;
      })
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("fill", (d: Node) => {
        if (focusedNodeId) {
          return allowedNodeIds.has(d.id) ? 'currentColor' : '#bbb';
        }
        return 'currentColor';
      })
      .style("pointer-events", "none")
      .style("font-weight", "500")
      .style("text-shadow", "var(--background) 0 0 3px, var(--background) 0 0 2px")
      .attr("class", "node-label text-foreground")
      .attr("opacity", (d: Node) => {
        if (focusedNodeId) {
          return allowedNodeIds.has(d.id) ? 1 : 0.25;
        }
        return 1;
      });

    // Show node labels by default
    labels.style("opacity", 1);

    // Update node labels visibility based on zoom level
    function updateLabels() {
      const scale = transform.k
      labels.style("opacity", scale > 0.5 ? 1 : 0) // Показываем метки при меньшем уровне зума
    }

    // Настраиваем зум
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
        // Сохраняем текущую трансформацию в ref
        transformRef.current = event.transform;
      });

    // Применяем зум к SVG
    svg.call(zoom);
    
    // Если есть сохраненная трансформация, восстанавливаем её
    if (currentTransform) {
      svg.call(zoom.transform, currentTransform);
    }
    
    // Сохраняем симуляцию в ref
    simulationRef.current = simulation;
    
    // Click on background to clear preview and focus
    svg.on("click", () => {
      setPreviewNode(null);
      setFocusedNodeId(null);
      setHoveredLink(null);
    });

    // Update simulation on tick
    simulation.nodes(data.nodes).on("tick", () => {
      // Обновляем позиции линий связей с использованием путей
      link.attr("d", generateLinkPath);

      // Обновляем позиции невидимых линий для улучшения области взаимодействия
      container.selectAll(".link-hit-areas path")
        .attr("d", generateLinkPath);

      // Обновляем позиции узлов
      node.attr("transform", (d: Node) => `translate(${d.x},${d.y})`)
      
      // Обновляем позиции текстовых меток
      labels
        .attr("x", (d: Node) => d.x!)
        .attr("y", (d: Node) => {
          // Получаем значимость из разных источников данных
          const importance = d.originalData?.importance || d.val || 1;
          // Уменьшаем коэффициент с 2 до 0.8
          return d.y! + 15 + importance * importance * 0.8;
        })
    })

    // --- FIT TO SCREEN (вид сверху) ---
    // Функция для автоматического масштабирования графа
    function fitToScreen() {
      if (!svgRef.current || !data) return;
      const xs = data.nodes.map(n => n.x ?? 0);
      const ys = data.nodes.map(n => n.y ?? 0);
      if (xs.length === 0 || ys.length === 0) return;
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const graphWidth = maxX - minX;
      const graphHeight = maxY - minY;
      const svgWidth = width;
      const svgHeight = height;
      const padding = 40;
      const scale = Math.min(
        (svgWidth - 2 * padding) / (graphWidth || 1),
        (svgHeight - 2 * padding) / (graphHeight || 1),
        1
      );
      const tx = (svgWidth - scale * (minX + maxX)) / 2;
      const ty = (svgHeight - scale * (minY + maxY)) / 2;
      const svg = d3.select(svgRef.current);
      svg.transition().duration(600).call(
        zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      );
      transformRef.current = d3.zoomIdentity.translate(tx, ty).scale(scale);
    }

    // Ждём несколько тиков симуляции, чтобы координаты устаканились, и делаем fit-to-screen только если ещё не делали
    if (!hasFitToScreen) {
      setTimeout(() => {
        fitToScreen();
        setHasFitToScreen(true);
      }, 600);
    }

    // @ts-ignore - d3 typing issue
    simulation.force("link").links(links)
    
    // Проверяем, что ссылки корректно добавлены в симуляцию
    console.log("Actual links in simulation:", 
      // @ts-ignore - доступ к приватному полю для отладки
      (simulation.force("link") as any)?._links?.length || "unknown");

    // Drag functions
    function dragstarted(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      event.subject.fx = event.subject.x
      event.subject.fy = event.subject.y
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      event.subject.fx = event.x
      event.subject.fy = event.y
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, Node, Node>) {
      if (!event.active) simulation.alphaTarget(0)
      event.subject.fx = null
      event.subject.fy = null
    }

    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    }
  }, [data, width, height, onNodeClick, showNodePreview, showLinkPreview, hideLinkPreview, focusedNodeId, hasFitToScreen])

  return (
    <div className={cn("relative w-full h-full", className)}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="bg-transparent"
      />

      {/* Легенда справа в углу */}
      <div
        style={{
          position: 'absolute',
          top: 24,
          right: 24,
          zIndex: 20,
          background: 'rgba(255,255,255,0.85)',
          borderRadius: 12,
          boxShadow: '0 2px 12px 0 rgba(0,0,0,0.08)',
          padding: '16px 20px',
          minWidth: 170,
          fontSize: 14,
          color: '#222',
          pointerEvents: 'auto',
        }}
      >
        <div style={{fontWeight: 600, marginBottom: 8, letterSpacing: 0.2}}>Легенда</div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
          {/* Цвета и подписи */}
          <LegendItem color="#22c55e" label="Понятие" />
          <LegendItem color="#ef4444" label="Персона" />
          <LegendItem color="#3b82f6" label="Событие" />
          <LegendItem color="#06b6d4" label="Место" />
          <LegendItem color="#f59e0b" label="Организация" />
          <LegendItem color="#8b5cf6" label="Термин" />
          <LegendItem color="#ec4899" label="Процесс" />
          <LegendItem color="#14b8a6" label="Объект" />
          <LegendItem color="#f97316" label="Произведение" />
          <LegendItem color="#6366f1" label="Теория" />
        </div>
      </div>

      {previewNode && (
        <NodePreview
          node={previewNode}
          position={previewPosition}
          onClose={() => {
            setPreviewNode(null);
            setFocusedNodeId(null);
          }}
          onViewDetails={() => {
            if (onNodeClick) onNodeClick(previewNode);
            setPreviewNode(null);
            setFocusedNodeId(null);
          }}
        />
      )}

      {hoveredLink && (
        <div 
          className="absolute z-50 bg-popover text-popover-foreground shadow-md rounded-md p-3 min-w-[120px]"
          style={{
            left: `${hoverPosition.x + 10}px`,
            top: `${hoverPosition.y + 10}px`,
          }}
          onMouseEnter={handleMouseEnterPreview}
          onMouseLeave={handleMouseLeavePreview}
        >
          <div className="font-medium">{getRelationTypeLabel(hoveredLink.relation_type)}</div>
          {hoveredLink.originalData?.description && (
            <div className="text-sm opacity-80 mt-1">{hoveredLink.originalData.description}</div>
          )}
        </div>
      )}
    </div>
  )
}

// Компонент для одного элемента легенды
function LegendItem({ color, label }: { color: string, label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{
        display: 'inline-block',
        width: 16,
        height: 16,
        borderRadius: '50%',
        background: color,
        border: '2px solid #fff',
        boxShadow: '0 0 2px #aaa',
        marginRight: 4,
      }} />
      <span>{label}</span>
    </div>
  )
}
