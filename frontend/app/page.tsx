"use client"

import React, { useState, useEffect, useRef, FormEvent, ChangeEvent } from "react"
import {
  ArrowRight,
  Search,
  MapIcon,
  MessageSquare,
  RefreshCw,
  X,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { Sidebar } from "@/components/sidebar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { D3Graph } from "@/components/d3-graph"
import { MindMap, type MindMapNode } from "@/components/mindmap"
import { NodeDetailPanel } from "@/components/node-detail-panel"
import { ChatPanel } from "@/components/chat-panel"
import { mindmapData } from "@/lib/sample-data"
import { graphApi } from "@/lib/api"
import { Progress } from "@/components/ui/progress"
import { AnimatePresence, motion } from "framer-motion"
import { MarkmapDisplay } from "@/components/markmap-display"

// Placeholder texts for the chat button
const chatPlaceholders = [
  "Что такое белок? ",
  "Расскажи про ДНК",
  "Объясни строение атома",
  "Как устроена клетка?",
  "Спроси меня о науке",
];

// Создаем интерфейс для фильтров
interface FilterState {
  subjects: string[];
  grades: string[];
  topics: string[];
  subtopics: string[];
}

// Создаем глобальный объект для событий фильтрации
export const filterEvents = {
  listeners: new Set<(filters: FilterState) => void>(),
  
  // Метод для подписки на событие изменения фильтров
  subscribe(callback: (filters: FilterState) => void) {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  },
  
  // Метод для уведомления о изменении фильтров
  notify(filters: FilterState) {
    this.listeners.forEach(callback => callback(filters));
  }
};

// Добавляем интерфейс MindMapGeneratedData
export interface MindMapGeneratedData {
  mindMapMarkdown: string;
  complexityLevel?: string;
}

// Создадим отдельный компонент для анимированных подсказок, чтобы изолировать его обновления
// Это предотвратит ререндер графа при изменении текста
const AnimatedChatPlaceholder = () => {
  const [currentPlaceholderIndex, setCurrentPlaceholderIndex] = useState(0);
  
  // Effect to cycle through placeholders
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentPlaceholderIndex((prevIndex) =>
        (prevIndex + 1) % chatPlaceholders.length
      );
    }, 4000); // Change every 4 seconds

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, []);
  
  return (
    <div className="flex-1 overflow-hidden">
      <AnimatePresence mode="wait">
        <motion.span
          key={currentPlaceholderIndex}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -5 }}
          transition={{ duration: 0.3 }}
          className="block whitespace-nowrap"
        >
          {chatPlaceholders[currentPlaceholderIndex]}
        </motion.span>
      </AnimatePresence>
    </div>
  );
};

// Функция для анализа глубины mindmap (markdown)
function getMindMapDepth(markdown: string): number {
  const lines = markdown.split('\n').map(l => l.trim());
  let maxDepth = 0;
  for (const line of lines) {
    const match = line.match(/^(#+)\s+/);
    if (match) {
      const depth = match[1].length;
      if (depth > maxDepth) maxDepth = depth;
    }
  }
  return maxDepth;
}

// Функция для фильтрации markdown: оставляет только строки-заголовки
function sanitizeMindMapMarkdown(markdown: string): string {
  return markdown
    .split('\n')
    .map(line => line.trim())
    .filter(line => /^#+\s+/.test(line))
    .join('\n');
}

export default function KnowledgeGraph() {
  const [activeTab, setActiveTab] = useState(() => {
    // Check if we're in a browser environment
    if (typeof window !== 'undefined') {
      // Try to get the saved tab from localStorage
      const savedTab = localStorage.getItem('activeTab');
      // Return the saved tab if it exists, otherwise default to "graph"
      return savedTab || "graph";
    }
    return "graph";
  });
  const [selectedNode, setSelectedNode] = useState<any>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [userQuery, setUserQuery] = useState("")
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [currentQuery, setCurrentQuery] = useState("")
  const [isSearchVisible, setIsSearchVisible] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [generatedMindMap, setGeneratedMindMap] = useState<string | null>(null)
  const [sanitizedMindMap, setSanitizedMindMap] = useState<string | null>(null)
  const [showMindMapNotification, setShowMindMapNotification] = useState(false)
  const [isGeneratingMindMap, setIsGeneratingMindMap] = useState(false)
  const [generationProgress, setGenerationProgress] = useState(0)
  const [graphDataState, setGraphDataState] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isFiltering, setIsFiltering] = useState(false)
  const [mindMapError, setMindMapError] = useState<string | null>(null)
  const [lastMindMapRequest, setLastMindMapRequest] = useState<any>(null)
  const [currentComplexityLevel, setCurrentComplexityLevel] = useState<string | undefined>(undefined)
  const inputRef = useRef<{ sendMessage: (text: string) => void }>(null)
  const [isChatTyping, setIsChatTyping] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  // Эффект для санитизации mindmap при изменении генерируемых данных
  useEffect(() => {
    if (generatedMindMap) {
      setSanitizedMindMap(sanitizeMindMapMarkdown(generatedMindMap));
    } else {
      setSanitizedMindMap(null);
    }
  }, [generatedMindMap]);

  // Функция для загрузки графа с применением фильтров
  const loadGraphWithFilters = async (filters?: FilterState) => {
    try {
      setIsFiltering(true)
      
      let data;
      
      if (filters && (
          filters.subjects.length > 0 || 
          filters.grades.length > 0 || 
          filters.topics.length > 0 || 
          filters.subtopics.length > 0
        )) {
        // Если есть фильтры, используем endpoint для фильтрации
        const response = await fetch('/api/graph/filter', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(filters)
        });
        
        if (!response.ok) {
          throw new Error(`Ошибка запроса: ${response.status}`);
        }
        
        data = await response.json();
        console.log("Получены отфильтрованные данные:", data);
      } else {
        // Если нет фильтров, загружаем полный граф
        data = await graphApi.getGraphData();
      }
      
      // Преобразуем данные в формат, который принимает D3Graph (та же логика, что и в исходном useEffect)
      const nodesMap = new Map();
        
      // Преобразуем узлы
      const nodes = Array.isArray(data.nodes) 
        ? data.nodes.map((node: any) => {
            const formattedNode = {
              id: node.id,
              label: node.title || node.name,
              group: getGroupFromNodeType(node.type),
              val: node.importance || 5,
              // Сохраняем все оригинальные данные
              originalData: { ...node },
              // Дублируем важные поля на верхнем уровне для совместимости
              title: node.title,
              description: node.description,
              summary: node.summary,
              type: node.type,
              importance: node.importance
            };
            
            // Добавляем в map для быстрого поиска
            nodesMap.set(node.id, formattedNode);
            
            return formattedNode;
          }) 
        : [];
      
      // Преобразуем связи, проверяя, что узлы существуют
      const links = Array.isArray(data.edges) 
        ? data.edges
            .filter((edge: any) => {
              const sourceExists = nodesMap.has(edge.source_id);
              const targetExists = nodesMap.has(edge.target_id);
              
              if (!sourceExists) console.warn(`Source node ${edge.source_id} not found`);
              if (!targetExists) console.warn(`Target node ${edge.target_id} not found`);
              
              return sourceExists && targetExists;
            })
            .map((edge: any) => ({
              source: edge.source_id,
              target: edge.target_id,
              value: edge.strength || 1,
              originalData: edge // Сохраняем оригинальные данные
            })) 
        : [];
      
      const formattedData = { nodes, links };
      
      console.log(`Nodes: ${nodes.length}, Links: ${links.length}`);
      
      setGraphDataState(formattedData);
      setError(null);
    } catch (err) {
      console.error("Failed to load graph data:", err);
      setError("Failed to load graph data. Please try again later.");
    } finally {
      setLoading(false);
      setIsFiltering(false);
    }
  };

  // Функция для определения группы на основе типа узла
  function getGroupFromNodeType(type: string): number {
    switch(type) {
      case 'concept': return 0;
      case 'person': return 1;
      case 'event': return 2;
      case 'location': return 3;
      case 'organization': return 4;
      case 'term': return 5;
      default: return 0;
    }
  }

  // Fetch graph data from API when component mounts
  useEffect(() => {
    loadGraphWithFilters();

    // Подписываемся на событие изменения фильтров
    const unsubscribe = filterEvents.subscribe((filters) => {
      console.log("Получено событие изменения фильтров:", filters);
      loadGraphWithFilters(filters);
    });

    // Отписываемся при размонтировании компонента
    return () => unsubscribe();
  }, []);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    updateDimensions()
    window.addEventListener("resize", updateDimensions)
    return () => window.removeEventListener("resize", updateDimensions)
  }, [])

  // Save activeTab to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('activeTab', activeTab);
  }, [activeTab]);

  // Обработка поиска
  const handleSearch = async (value: string) => {
    if (!value.trim()) return;
    
    try {
      const results = await graphApi.searchNodes(value);
      console.log("Search results:", results);
      
      // Если найден только один результат, выбираем его
      if (results.results && results.results.length === 1) {
        const nodeId = results.results[0].id;
        await handleNodeDetail(nodeId);
      }
      
      // Здесь можно реализовать показ выпадающего списка с результатами
      // Например, устанавливать state с результатами поиска
      
    } catch (error) {
      console.error("Error searching nodes:", error);
    }
  };
  
  // Получение деталей узла по ID
  const handleNodeDetail = async (nodeId: string) => {
    try {
      const nodeData = await graphApi.getNode(nodeId);
      setSelectedNode(nodeData);
      setIsPanelOpen(true);
    } catch (error) {
      console.error("Error fetching node details:", error);
    }
  };

  const handleNodeClick = (node: any) => {
    // Если у узла есть оригинальные данные, используем их
    if (node.originalData) {
      setSelectedNode(node.originalData);
      setIsPanelOpen(true);
    } else {
      // Иначе загружаем полные данные по API
      handleNodeDetail(node.id);
    }
  }

  const handleQuerySubmit = (e: FormEvent) => {
    e.preventDefault()
    if (userQuery.trim()) {
      setCurrentQuery("") // Сначала сбрасываем текущий запрос
      setTimeout(() => {
        // Затем устанавливаем новый с небольшой задержкой
        setCurrentQuery(userQuery)
        setIsChatOpen(true)
        setUserQuery("")
        // Reset generated mind map when starting a new query
        setGeneratedMindMap(null)
        setShowMindMapNotification(false)
        setIsGeneratingMindMap(true)

        // Запускаем имитацию прогресса, хотя на самом деле это происходит асинхронно
        setGenerationProgress(0)
        const progressInterval = setInterval(() => {
          setGenerationProgress((prev: number) => {
            if (prev >= 95) {
              clearInterval(progressInterval)
              return prev
            }
            return prev + Math.random() * 15
          })
        }, 1000)
      }, 10)
    }
  }

  // Функция для расширения узла при клике на него
  const handleNodeExpand = (nodeName: string) => {
    if (!isChatOpen) {
      setIsChatOpen(true);
    }
    // Формируем запрос на расширение узла
    const expandQuery = `Расскажи подробнее о "${nodeName}"`;
    // Диспатчим событие для ChatPanel, чтобы корректно обработать расширение узла
    document.dispatchEvent(new CustomEvent('expandNode', { detail: { query: expandQuery } }));
  };

  const handleMindMapGenerated = (data: MindMapGeneratedData | MindMapNode) => {
    console.log("Mind map generated");
    // Use batch updates to prevent multiple re-renders
    if ('mindMapMarkdown' in data && typeof data.mindMapMarkdown === 'string') {
      console.log("Setting markdown mind map");
      // Group all state updates together to minimize re-renders
      const updateStates = () => {
        setGeneratedMindMap(data.mindMapMarkdown);
        if ('complexityLevel' in data) {
          setCurrentComplexityLevel(data.complexityLevel as string);
        }
        setIsGeneratingMindMap(false);
        setGenerationProgress(100);
        const isNotEmpty = data.mindMapMarkdown.trim() !== "";
        setShowMindMapNotification(isNotEmpty);
        // Only change tab if we're not already on "mindmap" and карта не пустая
        if (activeTab !== "mindmap" && isNotEmpty) {
          setActiveTab("mindmap");
        }
      };
      // Apply all updates at once
      updateStates();
      // Use a ref to track if component is still mounted before hiding notification
      const timeoutId = setTimeout(() => setShowMindMapNotification(false), 5000);
      return () => clearTimeout(timeoutId); // Cleanup timeout if component unmounts
    } 
    else {
      console.log("Setting object mind map");
      // Group all state updates together for the alternate format too
      const updateStates = () => {
        setGeneratedMindMap("# " + (data as MindMapNode).name);
        setIsGeneratingMindMap(false);
        setGenerationProgress(100);
        setShowMindMapNotification(true);
        // Only change tab if we're not already on "mindmap"
        if (activeTab !== "mindmap") {
          setActiveTab("mindmap");
        }
      };
      // Apply all updates at once
      updateStates();
      const timeoutId = setTimeout(() => setShowMindMapNotification(false), 5000);
      return () => clearTimeout(timeoutId); // Cleanup timeout if component unmounts
    }
  };

  // Функция для обработки ошибок генерации mindmap
  const handleMindMapError = (errorMessage: string, failedRequestParams: any) => {
    console.error("Mind map generation failed:", errorMessage, failedRequestParams);
    setMindMapError(errorMessage);
    setLastMindMapRequest(failedRequestParams);
    setIsGeneratingMindMap(false);
    setGenerationProgress(0);
  };

  // Функция повторной попытки генерации mindmap
  const handleRetryMindMap = async () => {
    if (!lastMindMapRequest) return;
    
    console.log("Retrying mind map generation with params:", lastMindMapRequest);
    setMindMapError(null);
    setIsGeneratingMindMap(true);
    setGenerationProgress(30);
    
    const paramsToRetry = { ...lastMindMapRequest };
    setLastMindMapRequest(null);

    try {
      const response = await fetch("/api/mindmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(paramsToRetry),
      });
      
      setGenerationProgress(70);
      const data = await response.json();
      
      if (!response.ok || data.error) {
        const apiErrorMessage = data?.error?.message || data?.error || `HTTP error! status: ${response.status}`;
        throw new Error(apiErrorMessage);
      }
      
      if (data.mindMapMarkdown && typeof data.mindMapMarkdown === 'string') {
        console.log("Mind map retry successful");
        handleMindMapGenerated({ mindMapMarkdown: data.mindMapMarkdown });
      } else {
        throw new Error("API success but no mindMapMarkdown field received on retry");
      }
    } catch (error: any) {
      console.error("Error retrying mind map generation:", error);
      setLastMindMapRequest(paramsToRetry);
      setMindMapError(`Ошибка при повторной попытке: ${error.message || "Неизвестная ошибка"}`);
      setIsGeneratingMindMap(false);
      setGenerationProgress(0);
    }
  };

  // Функция отмены ошибки mindmap
  const dismissMindMapError = () => {
    setMindMapError(null);
    setLastMindMapRequest(null);
  };

  const switchToMindMap = () => {
    setActiveTab("mindmap")
  }

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen)
  }

  // Функция для перехода из карточки ноды в mindmap с автозапросом
  const handleExplainNode = (nodeTitle: string, nodeContext?: string) => {
    const query = `Расскажи мне про ${nodeTitle}${nodeContext ? ` с контекстом: ${nodeContext}` : ''}`;
    setIsChatOpen(true);
    setIsPanelOpen(false);
    setTimeout(() => {
      inputRef.current?.sendMessage(query);
    }, 100);
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar - только на графе и если открыто */}
      {activeTab === "graph" && isSidebarOpen && (
        <Sidebar onClose={() => setIsSidebarOpen(false)} />
      )}
      {/* Кнопка открытия Sidebar */}
      {activeTab === "graph" && !isSidebarOpen && (
        <button
          className="fixed top-24 left-2 z-30 bg-background border border-border rounded-full shadow p-1 hover:bg-accent transition"
          onClick={() => setIsSidebarOpen(true)}
          aria-label="Открыть меню"
        >
          <ChevronRight className="w-6 h-6" />
        </button>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <header className="p-4 border-b border-border bg-background flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-10 h-10 rounded-md bg-foreground flex items-center justify-center mr-3">
              <svg width="20" height="20" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="4" cy="4" r="2.5" fill="currentColor" className="text-background" />
                <circle cx="10" cy="4" r="2.5" fill="currentColor" className="text-background" />
                <circle cx="4" cy="10" r="2.5" fill="currentColor" className="text-background" />
                <circle cx="10" cy="10" r="2.5" fill="currentColor" className="text-background" />
              </svg>
            </div>
            <div>
              <h1 className="font-medium text-lg">{activeTab === "graph" ? "Граф знаний" : "Mind Map"}</h1>
              <p className="text-xs text-muted-foreground">
                {activeTab === "graph"
                  ? "Визуализация связей между понятиями"
                  : "Структурированное представление знаний"}
              </p>
            </div>
            
            {/* Добавляем индикатор уровня сложности, если есть mindmap */}
            {activeTab === "mindmap" && currentComplexityLevel && (
              <div className="ml-4 px-2 py-1 bg-primary/10 text-primary text-xs rounded-md">
                {currentComplexityLevel === "школьный" ? "Школьный уровень" : 
                 currentComplexityLevel === "университетский" ? "Университетский уровень" : 
                 currentComplexityLevel === "продвинутый" ? "Продвинутый уровень" : 
                 currentComplexityLevel}
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <Button
              variant={activeTab === "graph" ? "default" : "outline"}
              onClick={() => setActiveTab("graph")}
              className="h-9"
            >
              Граф Знаний
            </Button>
            <Button
              variant={activeTab === "mindmap" ? "default" : "outline"}
              onClick={() => setActiveTab("mindmap")}
              className="h-9 relative"
            >
              Mind Map
              {showMindMapNotification && activeTab !== "mindmap" && (
                <span className="absolute -top-1 -right-1 flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
                </span>
              )}
            </Button>
          </div>
        </header>

        <main className="flex-1 relative bg-background grid-pattern-light dark:grid-pattern-dark" ref={containerRef}>
          {/* Graph controls - скрываем поисковую строку */}
          {isSearchVisible && (
            <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
              <Card className="w-64 shadow-sm">
                <CardContent className="p-3">
                  <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="search"
                      placeholder="Поиск узла..."
                      className="pl-9 h-9"
                      value={searchQuery}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Loading state */}
          {(loading || isFiltering) && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80">
              <div className="text-center">
                <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-sm text-muted-foreground">
                  {isFiltering ? "Применение фильтров..." : "Загрузка данных графа..."}
                </p>
              </div>
            </div>
          )}

          {/* Error state */}
          {error && !loading && !isFiltering && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80">
              <div className="text-center max-w-md mx-auto p-6 bg-card border border-border rounded-lg shadow-sm">
                <span className="text-destructive text-lg mb-4 inline-block">⚠️</span>
                <h3 className="text-lg font-medium mb-2">Ошибка загрузки</h3>
                <p className="text-sm text-muted-foreground mb-4">{error}</p>
                <Button onClick={() => loadGraphWithFilters()}>Повторить попытку</Button>
              </div>
            </div>
          )}

          {/* Graph visualization */}
          {activeTab === "graph" && !loading && !error && (
            <div className="absolute inset-0 animate-fade-in">
              <D3Graph
                data={graphDataState}
                width={dimensions.width}
                height={dimensions.height}
                onNodeClick={handleNodeClick}
              />
            </div>
          )}

          {activeTab === "mindmap" && generatedMindMap && (
            <div className="absolute inset-0 z-10">
              {/* Warnings about shallow mind map */}
              {getMindMapDepth(generatedMindMap) <= 1 && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-yellow-100 text-yellow-900 border border-yellow-300 rounded px-4 py-2 shadow-md max-w-xl text-center">
                  <b>Внимание:</b> Карта знаний содержит только один уровень. Попробуйте задать более конкретный вопрос или попросите раскрыть детали для получения более глубокой mind map.
                </div>
              )}
              
              {sanitizedMindMap && (
                <MarkmapDisplay
                  markdown={sanitizedMindMap}
                  className="w-full h-full bg-transparent"
                  onNodeExpand={handleNodeExpand}
                />
              )}
            </div>
          )}

          {/* Placeholder if no map */}
          {activeTab === "mindmap" && !generatedMindMap && (
            <div className="absolute inset-0 flex items-center justify-center flex-col gap-4 animate-fade-in">
              <div className="text-center max-w-md p-6">
                <MapIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-xl font-medium mb-2">Начните чат для создания карты</h3>
                <p className="text-muted-foreground mb-4">
                  Нажмите кнопку чата, чтобы задать вопрос и сгенерировать карту знаний.
                </p>
                <Button onClick={toggleChat}>Открыть чат</Button>
              </div>
            </div>
          )}

          {/* Mind map generation progress */}
          {isGeneratingMindMap && (
            <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-card border border-border rounded-lg shadow-lg p-4 animate-slide-in z-50 w-80">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-primary/10 rounded-full p-2">
                  <MapIcon className="h-6 w-6 text-primary animate-pulse" />
                </div>
                <div>
                  <h4 className="font-medium text-sm">Создание карты знаний</h4>
                  <p className="text-xs text-muted-foreground">Пожалуйста, подождите...</p>
                </div>
              </div>
              <Progress value={generationProgress} className="h-2" />
            </div>
          )}

          {/* Mind map notification */}
          {showMindMapNotification && activeTab !== "mindmap" && (
            <div className="fixed bottom-20 right-4 bg-card border border-border rounded-lg shadow-lg p-4 animate-slide-in z-50">
              <div className="flex items-center gap-3">
                <div className="bg-primary/10 rounded-full p-2">
                  <MapIcon className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h4 className="font-medium text-sm">Карта знаний создана</h4>
                  <p className="text-xs text-muted-foreground">Переключитесь на вкладку Mind Map для просмотра</p>
                </div>
              </div>
              <div className="mt-3 flex justify-end">
                <Button size="sm" onClick={switchToMindMap}>
                  Посмотреть
                </Button>
              </div>
            </div>
          )}

          {/* Node detail panel */}
          <NodeDetailPanel node={selectedNode} isOpen={isPanelOpen} onClose={() => setIsPanelOpen(false)} onExplain={handleExplainNode} />

          {/* Chat panel */}
          <ChatPanel
            isOpen={isChatOpen}
            onClose={() => setIsChatOpen(false)}
            onMindMapGenerated={(data) => {
              if (data && typeof data === 'object' && 'mindMapMarkdown' in data && typeof data.mindMapMarkdown === 'string') {
                // Передаем уровень сложности, если он есть в ответе чата
                if ('complexityLevel' in data) {
                  handleMindMapGenerated({
                    mindMapMarkdown: data.mindMapMarkdown,
                    complexityLevel: data.complexityLevel as string
                  });
                } else {
                  handleMindMapGenerated(data);
                }
              } else {
                console.error("Received unexpected data format for mind map:", data);
                handleMindMapError("Неверный формат данных от API карты", {
                  latestMessage: { id: 'error-placeholder', role: 'system', content: 'Error context unavailable'},
                  currentMindMapMarkdown: generatedMindMap
                });
              }
            }}
            onMindMapError={handleMindMapError}
            onSwitchToMindMap={switchToMindMap}
            ref={inputRef}
            onTypingChange={setIsChatTyping}
          />

          {/* Dock at the bottom of the screen (new) */}
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20">
            <div className="flex items-center gap-1 px-3 py-1.5 bg-background/80 backdrop-blur-sm border border-border rounded-full shadow-lg">
              {/* Chat Button/Placeholder */}
              <TooltipProvider delayDuration={100}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      className="h-7 px-3 w-64 text-sm text-muted-foreground hover:text-foreground hover:bg-accent/60 rounded-full relative overflow-hidden flex items-center gap-1 justify-start border border-transparent hover:border-primary/30 transition-colors duration-200"
                      onClick={toggleChat}
                    >
                      {/* Add icon before the text */}
                      <MessageSquare className="h-5 w-5 flex-shrink-0 mr-0.5" />
                      <AnimatedChatPlaceholder />
                      {/* Typing indicator */}
                      {isChatTyping && !isChatOpen && (
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-black opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-black"></span>
                        </span>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{isChatOpen ? "Закрыть чат" : "Открыть чат"}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          {/* Добавим отображение ошибки и возможность повторного запроса */}
          {mindMapError && (
            <div className="fixed top-20 left-1/2 transform -translate-x-1/2 w-auto max-w-2xl z-50 shadow-lg pointer-events-auto">
              <div className="bg-destructive/10 border border-destructive text-destructive p-4 rounded-lg">
                <div className="flex items-center justify-between gap-4">
                  <span className="flex-1 break-words mr-2">{mindMapError}</span>
                  <div className="flex gap-2 flex-shrink-0">
                    {lastMindMapRequest && (
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={handleRetryMindMap} 
                        className="h-7 px-2 shrink-0"
                      >
                        <RefreshCw className="h-3.5 w-3.5 mr-1" />
                        Повторить
                      </Button>
                    )}
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      onClick={dismissMindMapError} 
                      className="h-7 w-7 shrink-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>

      </div>
    </div>
  )
}
