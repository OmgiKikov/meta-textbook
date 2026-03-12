"use client"
import { useState, useEffect, useCallback } from "react"
import { X, ArrowRight, Share2, Download, Bookmark, BookmarkCheck, ExternalLink, ChevronDown, ChevronUp, MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"
import { graphApi } from "@/lib/api"

interface NodeDetailPanelProps {
  node: any
  isOpen: boolean
  onClose: () => void
  className?: string
  onExplain?: (nodeTitle: string, nodeContext?: string) => void
}

export function NodeDetailPanel({ node, isOpen, onClose, className, onExplain }: NodeDetailPanelProps) {
  const [activeTab, setActiveTab] = useState("overview")
  const [isBookmarked, setIsBookmarked] = useState(false)
  const [isVisible, setIsVisible] = useState(false)
  const [nodeDetails, setNodeDetails] = useState<any>(null)
  const [relatedNodes, setRelatedNodes] = useState<any[]>([])
  const [questions, setQuestions] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [questionsLoading, setQuestionsLoading] = useState(false)
  const [edgesData, setEdgesData] = useState<any[]>([])
  const [edgesLoading, setEdgesLoading] = useState(false)
  const [expandedBullets, setExpandedBullets] = useState<{ [key: string]: boolean }>({})
  const [bulletContexts, setBulletContexts] = useState<{ [key: string]: string }>({})
  const [loadingContexts, setLoadingContexts] = useState<{ [key: string]: boolean }>({})
  const router = useRouter()

  // Animate in when panel opens
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => {
        setIsVisible(true)
      }, 50)
      return () => clearTimeout(timer)
    } else {
      setIsVisible(false)
    }
  }, [isOpen])

  // Fetch full node details when node changes
  useEffect(() => {
    if (node && node.id) {
      // Отладочный вывод всей доступной информации о ноде
      console.log("Node data:", node);
      console.log("Node metadata:", 
        node.meta || 
        node.metadata || 
        node.originalData?.meta || 
        node.originalData?.metadata || 
        {}
      );
      
      // Сбрасываем позицию прокрутки всех вкладок при смене узла
      // Находим все контейнеры с прокруткой и сбрасываем их позицию
      document.querySelectorAll('.overflow-y-auto').forEach(el => {
        (el as HTMLElement).scrollTop = 0;
      });
      
      // Set loading state
      setLoading(true);
      
      // If we already have full node details (originalData), use that
      if (node.originalData) {
        console.log("Using originalData:", node.originalData);
        setNodeDetails(node.originalData);
        setLoading(false);
      } else {
        // Otherwise fetch from API
        const fetchNodeDetails = async () => {
          try {
            const details = await graphApi.getNode(node.id);
            console.log("API response details:", details);
            setNodeDetails(details);
          } catch (error) {
            console.error("Failed to fetch node details:", error);
          } finally {
            setLoading(false);
          }
        };
        
        fetchNodeDetails();
      }
      
      // Очищаем вопросы при смене узла
      setQuestions([]);
      
      // Загружаем связи узла через API
      setEdgesLoading(true);
      const fetchNodeEdges = async () => {
        try {
          // Подробный вывод всех метаданных узла для отладки
          console.log("Все данные узла:", node);
          console.log("Метаданные узла:", node.meta);
          console.log("Оригинальные данные узла:", node.originalData);
          
          if (node.originalData?.meta) {
            if (Array.isArray(node.originalData.meta)) {
              console.log("Meta в originalData (массив):", node.originalData.meta);
            } else {
              console.log("Meta в originalData (объект):", node.originalData.meta);
            }
          }
          
          // Более глубокое и гибкое извлечение предмета из метаданных
          let subject = null;
          
          // Проверяем все возможные пути к предмету в данных
          if (node.meta?.subject) {
            subject = node.meta.subject;
            console.log("Предмет из node.meta.subject:", subject);
          } else if (Array.isArray(node.meta) && node.meta.length > 0 && node.meta[0].subject) {
            subject = node.meta[0].subject;
            console.log("Предмет из node.meta[0].subject:", subject);
          } else if (node.originalData?.meta?.subject) {
            subject = node.originalData.meta.subject;
            console.log("Предмет из node.originalData.meta.subject:", subject);
          } else if (Array.isArray(node.originalData?.meta) && node.originalData.meta.length > 0) {
            subject = node.originalData.meta[0].subject;
            console.log("Предмет из node.originalData.meta[0].subject:", subject);
          } else if (node.id && (node.id.includes('history') || node.id.includes('истор'))) {
            subject = "История России";
            console.log("Предмет определен по ID узла (history):", subject);
          } else if (node.id && (node.id.includes('physics') || node.id.includes('физи'))) {
            subject = "Физика";
            console.log("Предмет определен по ID узла (physics):", subject);
          }
          
          // Если и это не помогло, пробуем извлечь предмет из названия узла или других данных
          if (!subject && getNodeTitle()) {
            const title = getNodeTitle().toLowerCase();
            if (title.includes('истор')) {
              subject = "История России";
              console.log("Предмет определен по названию узла (история):", subject);
            } else if (title.includes('физи') || title.includes('сила') || title.includes('энерг') || title.includes('механи')) {
              subject = "Физика";
              console.log("Предмет определен по названию узла (физика):", subject);
            } else if (title.includes('биолог') || title.includes('клетк') || title.includes('организм')) {
              subject = "Биология";
              console.log("Предмет определен по названию узла (биология):", subject);
            }
          }
          
          // Если всё ещё нет предмета, проверяем контекстные данные
          if (!subject) {
            const contextData = getContextData();
            if (contextData.subjects.length > 0) {
              subject = contextData.subjects[0];
              console.log("Предмет из контекстных данных:", subject);
            }
          }
          
          console.log("Итоговый предмет узла для загрузки связей:", subject);
          
          // Передаем предмет в API (даже если он null или undefined)
          const edgesData = await graphApi.getNodeEdges(node.id, subject);
          console.log("Получены связи узла:", edgesData);
          if (edgesData && edgesData.edges) {
            setRelatedNodes(edgesData.edges);
          }
        } catch (error) {
          console.error("Ошибка при загрузке связей:", error);
        } finally {
          setEdgesLoading(false);
        }
      };
      
      fetchNodeEdges();
    }
  }, [node]);

  // Функция загрузки связей из файла
  const loadEdgesFromFile = useCallback(async () => {
    try {
      console.log("Начинаем загрузку связей из файла...");
      const response = await fetch('/api/graph/data');
      if (!response.ok) {
        throw new Error(`Не удалось загрузить данные графа: ${response.statusText}`);
      }
      const graphData = await response.json();
      console.log("Загружены данные графа, связи:", graphData.edges?.length);
      
      // Устанавливаем данные связей из полученных данных графа
      if (graphData.edges) {
        setEdgesData(graphData.edges);
      } else {
        console.warn("В данных графа отсутствуют связи");
        setEdgesData([]);
      }
    } catch (error) {
      console.error("Ошибка при загрузке связей:", error);
    }
  }, []);

  // Загрузка связей при монтировании компонента
  useEffect(() => {
    loadEdgesFromFile();
  }, [loadEdgesFromFile]);

  if (!node) return null

  // Get node type from node details or node itself
  const getNodeType = () => {
    if (!node) return "unknown";
    
    // Проверяем разные источники типа
    const nodeType = nodeDetails?.type || 
           nodeDetails?.metadata?.type ||
           node.originalData?.type ||
           node.originalData?.meta?.type ||
           node.type || 
           node.group || 
           (node.id?.includes("concept") ? "concept" : 
            node.id?.includes("person") ? "person" : 
            node.id?.includes("event") ? "event" : "unknown");
    
    console.log("Node type:", nodeType);
    return nodeType;
  }

  // Get image based on node type or group
  const getNodeImage = () => {
    if (!node) return "/placeholder.svg?height=200&width=400";

    // Получаем ID узла
    const nodeId = node.id || node.originalData?.id || "";
    const nodeType = getNodeType();
    
    // Используем API для получения изображения
    return `/api/node-image?id=${encodeURIComponent(nodeId)}&type=${encodeURIComponent(nodeType)}`;
  }

  // Get node title (use the most reliable source)
  const getNodeTitle = () => {
    return nodeDetails?.title || node.label || node.name || node.title || "Узел знаний";
  }

  // Get node definition/description
  const getNodeDefinition = () => {
    // If we have a definition from API, use that
    if (nodeDetails?.description) {
      return nodeDetails.description;
    }

    if (!node) return "";

    const nodeId = node.id || "";
    const nodeName = getNodeTitle();

    if (nodeId.includes("atom") || nodeName.includes("Атом"))
      return "Атом — это наименьшая частица химического элемента, сохраняющая его свойства. Название происходит от греческого слова «atomos», что означает «неделимый». \n---\nСтруктура атома включает ядро (содержащее протоны и нейтроны) и электронное облако (область вокруг ядра, где движутся электроны).";
    if (nodeId.includes("dna") || nodeName.includes("ДНК"))
      return "ДНК (дезоксирибонуклеиновая кислота) — это молекула, которая содержит генетические инструкции для развития и функционирования всех живых организмов. \n---\nСтруктура ДНК представляет собой двойную спираль, состоящую из двух цепей нуклеотидов, связанных между собой водородными связями между комплементарными основаниями.";
    if (nodeId.includes("protein") || nodeName.includes("Белок"))
      return "Белки — это крупные биологические молекулы, состоящие из аминокислот. \n---\nОни являются основой всех живых организмов и выполняют множество важных функций, включая катализ биохимических реакций, транспорт веществ, структурную поддержку и иммунную защиту.";
    if (nodeId.includes("cell") || nodeName.includes("Клетка"))
      return "Клетка — это основная структурная и функциональная единица всех живых организмов. \n---\nЭто наименьшая единица жизни, способная к самостоятельному существованию и самовоспроизведению. \n---\nСуществует два основных типа клеток: прокариотические (простые клетки без ядра) и эукариотические (сложные клетки с ядром и органеллами).";

    return "Элемент структуры знаний, представляющий собой концепт или понятие в определенной области науки или знания. \n---\nСвязан с другими элементами через различные типы отношений, образуя сеть знаний.";
  }

  // Функция для загрузки контекста буллета
  const loadBulletContext = async (bulletId: string, nodeId: string, chunkId: string) => {
    // Проверяем, загружен ли уже контекст
    if (bulletContexts[bulletId]) return;
    
    // Устанавливаем состояние загрузки
    setLoadingContexts(prev => ({ ...prev, [bulletId]: true }));
    
    try {
      // Используем API-метод вместо прямого fetch
      const data = await graphApi.getBulletContext(nodeId, chunkId);
      
      if (data.success) {
        // Сохраняем полученный контекст
        setBulletContexts(prev => ({ 
          ...prev, 
          [bulletId]: data.context + (data.source ? `\n\nИсточник: ${data.source}` : '')
        }));
      } else {
        // Если произошла ошибка, сохраняем сообщение об ошибке
        setBulletContexts(prev => ({ 
          ...prev, 
          [bulletId]: `Не удалось загрузить контекст: ${data.context}` 
        }));
      }
    } catch (error) {
      console.error("Ошибка при загрузке контекста:", error);
      setBulletContexts(prev => ({ 
        ...prev, 
        [bulletId]: "Ошибка при загрузке контекста" 
      }));
    } finally {
      // Сбрасываем состояние загрузки
      setLoadingContexts(prev => ({ ...prev, [bulletId]: false }));
    }
  };

  // Функция для переключения состояния развернутости буллета
  const toggleBullet = (bulletId: string, nodeId: string, chunkId: string) => {
    // Переключаем состояние
    const newState = !expandedBullets[bulletId];
    setExpandedBullets(prev => ({ ...prev, [bulletId]: newState }));
    
    // Если буллет раскрывается и контекст еще не загружен, загружаем его
    if (newState && !bulletContexts[bulletId]) {
      loadBulletContext(bulletId, nodeId, chunkId);
    }
  };

  // Модифицированная функция для разделения текста на буллиты
  const formatTextAsBullets = (text: string | undefined) => {
    if (!text) return null;
    
    // Разделяем текст по маркеру \n---\n
    const parts = text.split("\n---\n");
    
    // Получаем ID узла
    const nodeId = node?.id || nodeDetails?.id || "";
    
    // Формируем список кликабельных буллетов (даже если только один)
    return (
      <ul className="list-disc pl-5 space-y-3">
        {parts.map((part, index) => {
          // Создаем уникальный ID для буллета
          const bulletId = `bullet-${nodeId}-${index}`;
          // chunkId для API запроса (считаем, что chunkId совпадает с индексом буллета)
          const chunkId = `${index}`;
          // Проверяем, развернут ли буллет
          const isExpanded = expandedBullets[bulletId] || false;
          // Проверяем статус загрузки контекста
          const isLoadingContext = loadingContexts[bulletId] || false;
          
          return (
            <li key={bulletId} className="text-sm">
              <div className="flex items-start gap-2">
                <div 
                  className="flex-1 cursor-pointer hover:text-foreground group"
                  onClick={() => toggleBullet(bulletId, nodeId, chunkId)}
                >
                  <div className="flex items-center gap-1">
                    <span className="text-muted-foreground group-hover:text-foreground">{part}</span>
                    {isExpanded ? (
                      <ChevronUp className="h-4 w-4 inline text-muted-foreground group-hover:text-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 inline text-muted-foreground group-hover:text-foreground" />
                    )}
                  </div>
                </div>
              </div>
              
              {/* Контейнер для контекста */}
              {isExpanded && (
                <div className="mt-2 ml-4 p-3 bg-muted/50 rounded-md text-xs border border-border">
                  {isLoadingContext ? (
                    <div className="flex items-center justify-center p-2">
                      <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin mr-2"></div>
                      <span>Загрузка контекста...</span>
                    </div>
                  ) : (
                    bulletContexts[bulletId] ? (
                      <div className="whitespace-pre-wrap">{bulletContexts[bulletId]}</div>
                    ) : (
                      <div>Контекст недоступен</div>
                    )
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    );
  };

  // Получаем контекстные данные из разных источников
  const getContextData = () => {
    // Выводим всю метаинформацию в консоль для отладки
    console.log("Getting context data...");
    
    // Получаем данные из разных источников
    const metaField = 
      node.meta || 
      node.metadata || 
      node.originalData?.meta || 
      node.originalData?.metadata ||
      nodeDetails?.meta ||
      nodeDetails?.metadata;
    
    console.log("Meta field found:", metaField);
    
    // Обработка в зависимости от типа metaField
    let meta: any = {};
    
    // Если meta - массив
    if (Array.isArray(metaField) && metaField.length > 0) {
      meta = metaField[0];
      console.log("Meta is array, using first element:", meta);
    } 
    // Если meta - объект
    else if (metaField && typeof metaField === 'object') {
      meta = metaField;
      console.log("Meta is object:", meta);
    }
    
    // Извлекаем все необходимые данные
    const result: Record<string, string[]> = {
      subjects: [],
      grades: [],
      topics: [],
      subtopics: [],
      importance: []
    };
    
    // Добавляем subject
    if (meta.subject) {
      result.subjects.push(meta.subject);
    } else if (meta.subjects && Array.isArray(meta.subjects)) {
      result.subjects = [...meta.subjects];
    }
    
    // Добавляем grade
    if (meta.grade) {
      result.grades.push(meta.grade);
    } else if (meta.grades && Array.isArray(meta.grades)) {
      result.grades = [...meta.grades];
    } else if (meta.class) {
      result.grades.push(meta.class);
    } else if (meta.classes && Array.isArray(meta.classes)) {
      result.grades = [...meta.classes];
    }
    
    // Добавляем topic
    if (meta.topic) {
      result.topics.push(meta.topic);
    } else if (meta.topics && Array.isArray(meta.topics)) {
      result.topics = [...meta.topics];
    }
    
    // Добавляем subtopic
    if (meta.subtopic) {
      result.subtopics.push(meta.subtopic);
    } else if (meta.subtopics && Array.isArray(meta.subtopics)) {
      result.subtopics = [...meta.subtopics];
    }
    
    // Добавляем importance
    if (meta.importance) {
      result.importance.push(meta.importance);
    }
    
    console.log("Extracted data:", result);
    
    return result;
  };

  // Get contexts based on node metadata or connections
  const getNodeContexts = () => {
    // If we have metadata from API, use that
    if (nodeDetails?.metadata && typeof nodeDetails.metadata === 'object') {
      const metadata = nodeDetails.metadata;
      const contexts = [];
      
      // Extract contexts from metadata fields
      if (metadata.subject) contexts.push(metadata.subject);
      if (metadata.topic) contexts.push(metadata.topic);
      if (metadata.subtopic) contexts.push(metadata.subtopic);
      if (metadata.grade) contexts.push(`${metadata.grade} класс`);
      
      // If we got some contexts, return them
      if (contexts.length > 0) return contexts;
    }

    if (!node) return [];

    const nodeId = node.id || "";
    const nodeName = getNodeTitle();

    // Фиксированные контексты для определенных типов узлов
    if (nodeId.includes("atom") || nodeName.includes("Атом"))
      return ["Физика", "Химия", "Строение вещества", "Квантовая механика", "Периодическая система"];
    if (nodeId.includes("dna") || nodeName.includes("ДНК"))
      return ["Биология", "Генетика", "Молекулярная биология", "Геномика", "Биотехнология"];
    if (nodeId.includes("protein") || nodeName.includes("Белок"))
      return ["Биохимия", "Молекулярная биология", "Физиология", "Протеомика", "Структурная биология"];
    if (nodeId.includes("cell") || nodeName.includes("Клетка"))
      return ["Цитология", "Биология", "Микробиология", "Гистология", "Клеточная биология"];

    // Для остальных узлов используем детерминированный подход на основе ID
    const allContexts = [
      "Физика",
      "Химия",
      "Биология",
      "Математика",
      "Информатика",
      "География",
      "История",
      "Астрономия",
      "Экология",
      "Геология",
      "Психология",
      "Социология",
      "Лингвистика",
    ];

    // Используем ID узла для детерминированного выбора контекстов
    let seed = 0;
    if (nodeId) {
      for (let i = 0; i < nodeId.length; i++) {
        seed += nodeId.charCodeAt(i);
      }
    } else {
      seed = Math.floor(Math.random() * 1000);
    }

    // Выбираем контексты на основе seed
    const selectedContexts = [];
    const numContexts = (seed % 4) + 3; // 3-6 контекстов

    for (let i = 0; i < numContexts; i++) {
      const index = (seed + i * 23) % allContexts.length;
      selectedContexts.push(allContexts[index]);
    }

    return selectedContexts;
  }

  // Get related nodes (connected nodes)
  const getRelatedNodes = () => {
    // Если данные о связях уже загружены через API
    if (relatedNodes && relatedNodes.length > 0) {
      console.log("Используем связи, загруженные через API:", relatedNodes);
      return relatedNodes.map((edge: any) => ({
        id: edge.relatedNodeId,
        name: edge.relatedNodeName,
        relationDescription: edge.description,
        relevance: edge.strength * 20, // преобразуем в процентный формат
        relation_type: edge.type,
        direction: edge.direction
      }));
    }

    // Если связи еще загружаются, показываем сообщение о загрузке
    if (edgesLoading) {
      console.log("Связи загружаются...");
      return [
        {
          id: "loading",
          name: "Загрузка связей...",
          relevance: 50
        }
      ];
    }

    console.log("Связи не найдены, используем API или фейковые данные");
    // Если данные из файла недоступны или не найдены связи для текущего узла, используем API или фиктивные данные
    if (nodeDetails?.connected_nodes && Array.isArray(nodeDetails.connected_nodes) && nodeDetails.connected_nodes.length > 0) {
      return nodeDetails.connected_nodes.map((connectedNode: any) => ({
        id: connectedNode.id,
        name: connectedNode.title || connectedNode.name || "Связанный узел",
        relevance: connectedNode.strength || 75,
      }));
    }

    if (!node) return [];

    const nodeId = node.id || "";

    // Use the node ID for deterministic selection of related nodes
    let seed = 0;
    if (nodeId) {
      for (let i = 0; i < nodeId.length; i++) {
        seed += nodeId.charCodeAt(i);
      }
    } else {
      seed = Math.floor(Math.random() * 1000);
    }

    const numNodes = (seed % 4) + 3; // 3-6 related nodes

    return Array.from({ length: numNodes }, (_, i) => {
      const relevance = ((seed + i * 17) % 60) + 40; // 40-99% relevance
      return {
        id: `related-${nodeId}-${i}`,
        name: `Связанный концепт ${i + 1}`,
        relevance: relevance,
        direction: i % 2 === 0 ? "incoming" : "outgoing" // чередуем для демонстрации
      };
    });
  }
  
  // Получаем входящие связи (направленные на текущий узел)
  const getIncomingNodes = () => {
    const allNodes = getRelatedNodes();
    if (!Array.isArray(allNodes)) return [];
    
    if (allNodes.length === 1 && allNodes[0].id === "loading") {
      return allNodes; // Возвращаем сообщение о загрузке
    }
    
    return allNodes
      .filter((node: {direction?: string}) => node.direction === "incoming" || node.direction === "in")
      .sort((a, b) => b.relevance - a.relevance); // Сортировка по убыванию relevance
  }
  
  // Получаем исходящие связи (направленные от текущего узла)
  const getOutgoingNodes = () => {
    const allNodes = getRelatedNodes();
    if (!Array.isArray(allNodes)) return [];
    
    if (allNodes.length === 1 && allNodes[0].id === "loading") {
      return []; // Если загрузка, для исходящих возвращаем пустой массив
    }
    
    return allNodes
      .filter((node: {direction?: string}) => node.direction === "outgoing" || node.direction === "out")
      .sort((a, b) => b.relevance - a.relevance); // Сортировка по убыванию relevance
  }

  // Get sources
  const getSources = () => {
    // If we have sources from API, use those
    if (nodeDetails?.sources && Array.isArray(nodeDetails.sources) && nodeDetails.sources.length > 0) {
      return nodeDetails.sources;
    }

    return [
      { title: "Учебник по общей биологии", author: "И.И. Иванов", year: 2020, type: "book" },
      { title: "Основы молекулярной биологии", author: "П.П. Петров", year: 2018, type: "article" },
      { title: "Современная биохимия", author: "С.С. Сидоров", year: 2021, type: "book" },
    ];
  }

  // Get questions about this node
  const getQuestions = () => {
    console.log("getQuestions вызвана, текущие вопросы:", questions);
    
    // Если вопросы загружены и массив не пустой, возвращаем их
    if (questions && Array.isArray(questions) && questions.length > 0) {
      console.log("Возвращаем загруженные вопросы:", questions);
      return questions;
    }
    
    console.log("Возвращаем шаблонные вопросы");
    // Fallback questions
    return [
      "Что такое " + getNodeTitle() + "?",
      "Каковы основные свойства " + getNodeTitle() + "?",
      "Как связан " + getNodeTitle() + " с другими понятиями?",
    ];
  }

  // Добавляем функцию для загрузки вопросов
  const loadQuestions = async () => {
    if (!node || !node.id) {
      console.log("Нет ID узла для загрузки вопросов");
      return;
    }
    
    // Переключаем состояние загрузки
    setQuestionsLoading(true);
    console.log("Начинаем загрузку вопросов для узла с ID:", node.id);
    
    try {
      console.log("Отправляем запрос на API:", `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/graph/generate_questions`);
      
      // Установим таймаут для запроса
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error("Timeout")), 5000);
      });
      
      // Создаем запрос
      const fetchPromise = graphApi.generateQuestions(node.id);
      
      // Ждем результат или таймаут (что наступит раньше)
      const response: any = await Promise.race([fetchPromise, timeoutPromise])
        .catch(error => {
          console.error("Ошибка загрузки вопросов (гонка):", error);
          return null;
        });
      
      console.log("Ответ от API:", response);
      
      if (response && response.questions && Array.isArray(response.questions)) {
        console.log("Получены вопросы от API:", response.questions);
        setQuestions(response.questions);
        console.log("Вопросы успешно сохранены в состояние");
      } else {
        console.warn("Ошибка или некорректный формат ответа от API", response);
        
        // Создаем шаблонные вопросы в качестве запасного варианта
        const fallbackQuestions = [
          `Что такое ${getNodeTitle()}?`,
          `Каковы основные свойства и характеристики ${getNodeTitle()}?`,
          `Как ${getNodeTitle()} связано с другими понятиями?`,
          `В чем значимость изучения ${getNodeTitle()}?`,
          `Какие примеры использования или проявления ${getNodeTitle()} существуют?`
        ];
        
        console.log("Используем шаблонные вопросы:", fallbackQuestions);
        setQuestions(fallbackQuestions);
      }
    } catch (error) {
      console.error("Неперехваченная ошибка при загрузке вопросов:", error);
      
      // Создаем шаблонные вопросы в случае ошибки
      const errorFallbackQuestions = [
        `Что такое ${getNodeTitle()}?`,
        `Каковы основные характеристики ${getNodeTitle()}?`,
        `Как ${getNodeTitle()} связано с другими концепциями?`,
      ];
      
      setQuestions(errorFallbackQuestions);
    } finally {
      setQuestionsLoading(false);
      console.log("Загрузка вопросов завершена");
    }
  };

  // Обработчик изменения активной вкладки
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    
    // Сбрасываем позицию прокрутки при переключении вкладки
    setTimeout(() => {
      document.querySelectorAll('.overflow-y-auto').forEach(el => {
        if (el.closest(`[data-state="active"]`)) {
          (el as HTMLElement).scrollTop = 0;
        }
      });
    }, 0);
    
    // Если переключились на вкладку с вопросами, загружаем их
    if (value === "questions") {
      // Очищаем текущие вопросы и загружаем заново
      setQuestions([]);
      setQuestionsLoading(true);
      
      console.log("Переключение на вкладку вопросов, загрузка вопросов для узла:", node?.id);
      loadQuestions();
    }
  };

  const handleNavigateToFullPage = () => {
    // In a real app, we would navigate to a dedicated page
    // router.push(`/node/${node.id}`)
    window.alert("В полной версии здесь будет переход на отдельную страницу с подробной информацией об узле.");
  }

  // Добавляем функцию обработки клика по кнопке "Открыть"
  const handleNodeClick = async (nodeId: string) => {
    console.log("Клик по узлу с ID:", nodeId);
    
    try {
      // Сбрасываем позицию прокрутки для всех контейнеров
      document.querySelectorAll('.overflow-y-auto').forEach(el => {
        (el as HTMLElement).scrollTop = 0;
      });
      
      // Загружаем данные нового узла
      setLoading(true);
      const newNodeData = await graphApi.getNode(nodeId);
      console.log("Загружены данные нового узла:", newNodeData);
      
      // Обновляем текущий узел, это приведет к перерисовке компонента
      // и запуску useEffect, который загрузит детали и связи
      const newNode = {
        ...newNodeData,
        id: nodeId,
        originalData: newNodeData // Сохраняем все данные в originalData
      };
      
      // Вместо закрытия панели и создания события, напрямую обновляем данные
      // Сбрасываем текущее состояние, чтобы оно обновилось для нового узла
      setNodeDetails(newNodeData);
      setRelatedNodes([]);
      setQuestions([]);
      
      // Загружаем связи для нового узла
      setEdgesLoading(true);
      try {
        // Подробный вывод метаданных нового узла
        console.log("Метаданные нового узла:", newNodeData.meta);
        
        // Более глубокое и гибкое извлечение предмета из метаданных
        let subject = null;
        
        // Проверяем все возможные пути к предмету в данных
        if (newNodeData.meta?.subject) {
          subject = newNodeData.meta.subject;
          console.log("Предмет из newNodeData.meta.subject:", subject);
        } else if (Array.isArray(newNodeData.meta) && newNodeData.meta.length > 0 && newNodeData.meta[0].subject) {
          subject = newNodeData.meta[0].subject;
          console.log("Предмет из newNodeData.meta[0].subject:", subject);
        } else if (nodeId && (nodeId.includes('history') || nodeId.includes('истор'))) {
          subject = "История России";
          console.log("Предмет определен по ID узла (history):", subject);
        } else if (nodeId && (nodeId.includes('physics') || nodeId.includes('физи'))) {
          subject = "Физика";
          console.log("Предмет определен по ID узла (physics):", subject);
        }
        
        // Если и это не помогло, пробуем извлечь предмет из названия узла
        if (!subject && (newNodeData.title || newNodeData.name)) {
          const title = (newNodeData.title || newNodeData.name).toLowerCase();
          if (title.includes('истор')) {
            subject = "История России";
            console.log("Предмет определен по названию узла (история):", subject);
          } else if (title.includes('физи') || title.includes('сила') || title.includes('энерг') || title.includes('механи')) {
            subject = "Физика";
            console.log("Предмет определен по названию узла (физика):", subject);
          } else if (title.includes('биолог') || title.includes('клетк') || title.includes('организм')) {
            subject = "Биология";
            console.log("Предмет определен по названию узла (биология):", subject);
          }
        }
        
        console.log("Итоговый предмет нового узла для загрузки связей:", subject);
        
        // Передаем предмет в API (даже если он null или undefined)
        const edgesData = await graphApi.getNodeEdges(nodeId, subject);
        console.log("Получены связи нового узла:", edgesData);
        if (edgesData && edgesData.edges) {
          setRelatedNodes(edgesData.edges);
        }
      } catch (error) {
        console.error("Ошибка при загрузке связей нового узла:", error);
      } finally {
        setEdgesLoading(false);
      }
      
      // Сбрасываем активную вкладку на "overview"
      setActiveTab("overview");
      
      // Вызываем событие, которое обновит родительский компонент
      // Это нужно, чтобы родительский компонент знал о смене узла
      if (window && window.parent) {
        window.parent.postMessage({
          type: 'NODE_SELECTED',
          nodeId: nodeId,
          node: newNode
        }, '*');
      }
      
      // Имитируем обновление текущего узла
      // Здесь мы создаем новый объект со всеми свойствами текущего узла,
      // но с данными нового узла
      const updatedNode = {
        ...node,
        id: nodeId,
        title: newNodeData.title || newNodeData.name || "Узел",
        label: newNodeData.title || newNodeData.name || "Узел",
        originalData: newNodeData
      };
      
      // Обновляем состояние node
      // Примечание: обновление node напрямую может не сработать, если node
      // передается из родительского компонента как пропс
      // В этом случае лучше создать локальную копию node
      if (node) {
        Object.assign(node, updatedNode);
      }
      
    } catch (error) {
      console.error("Ошибка при загрузке узла:", error);
      alert(`Ошибка при загрузке данных узла: ${nodeId}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity duration-300"
          style={{ opacity: isVisible ? 1 : 0 }}
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-screen w-[450px] bg-background border-l border-border shadow-lg transform transition-all duration-300 ease-in-out z-50 overflow-hidden",
          isOpen ? "translate-x-0" : "translate-x-full",
          isVisible ? "opacity-100" : "opacity-0",
          className,
        )}
      >
        {/* Loading state */}
        {loading && (
          <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-10">
            <div className="text-center">
              <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-sm text-muted-foreground">Загрузка данных...</p>
            </div>
          </div>
        )}
        
        {/* Header with image */}
        <div className="relative h-[250px] bg-muted">
          <img
            src={getNodeImage()}
            alt={getNodeTitle()}
            className="w-full h-full object-contain"
          />
        </div>

        {/* Action buttons */}
        <div className="flex items-center justify-between py-2 px-4 border-b border-border">
          <h2 className="text-black text-xl font-medium flex items-center gap-2">
            {getNodeTitle()}
          </h2>
          {onExplain && (
            <Button
              variant="default"
              className="flex items-center gap-2 px-4 py-2 rounded-lg shadow-md text-base font-semibold"
              onClick={() => onExplain(getNodeTitle(), getNodeDefinition())}
            >
              <MessageSquare className="w-5 h-5 mr-1" />
              Объясни
            </Button>
          )}
        </div>

        {/* Tabs */}
        <Tabs defaultValue={activeTab} onValueChange={handleTabChange} className="h-[calc(100%-355px)] overflow-hidden">
          <TabsList className="grid grid-cols-3 px-4 pt-4">
            <TabsTrigger value="overview">Обзор</TabsTrigger>
            <TabsTrigger value="relations">Связи</TabsTrigger>
            <TabsTrigger value="questions">Вопросы</TabsTrigger>
          </TabsList>

          {/* Overview tab */}
          <TabsContent value="overview" className="overflow-y-auto p-0 h-full">
            <div className="p-4 space-y-4 pb-16">
              {/* Definition */}
              <div>
                <h3 className="text-lg font-medium mb-2">Определение</h3>
                <p className="text-sm text-muted-foreground">{getNodeDefinition()}</p>
              </div>

               {/* Summary (if available) */}
               {(node.summary || node.originalData?.summary || nodeDetails?.summary) && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Краткое описание</h3>
                  {formatTextAsBullets(node.summary || node.originalData?.summary || nodeDetails?.summary)}
                </div>
              )}
              
              {/* Типы и контекстные данные */}
              <div>
                <h3 className="text-lg font-medium mb-2">Контекст и классификация</h3>
                <div className="flex gap-2 flex-wrap">
                  {/* Тип узла */}
                  <Badge variant="outline" className="bg-muted text-foreground border-border">
                    {getNodeType()}
                  </Badge>
                  
                  {/* Извлекаем и отображаем все контекстные данные */}
                  {(() => {
                    const contextData = getContextData();
                    
                    return (
                      <>
                        {/* Subjects */}
                        {contextData.subjects.map((subject, index) => (
                          <Badge 
                            key={`subject-${index}-${subject}`} 
                            variant="outline" 
                            className="bg-muted text-foreground border-border"
                          >
                            {subject}
                          </Badge>
                        ))}
                        
                        {/* Grades */}
                        {contextData.grades.map((grade, index) => (
                          <Badge 
                            key={`grade-${index}-${grade}`} 
                            variant="outline" 
                            className="bg-muted text-foreground border-border"
                          >
                            {typeof grade === 'number' || !isNaN(Number(grade)) ? `${grade} класс` : grade}
                          </Badge>
                        ))}
                        
                        {/* Topics */}
                        {contextData.topics.map((topic, index) => (
                          <Badge 
                            key={`topic-${index}-${topic}`} 
                            variant="outline" 
                            className="bg-muted text-foreground border-border"
                          >
                            {topic}
                          </Badge>
                        ))}
                        
                        {/* Subtopics */}
                        {contextData.subtopics.map((subtopic, index) => (
                          <Badge 
                            key={`subtopic-${index}-${subtopic}`} 
                            variant="outline" 
                            className="bg-muted text-foreground border-border"
                          >
                            {subtopic}
                          </Badge>
                        ))}
                        
                        {/* Importance */}
                        {contextData.importance.map((imp, index) => (
                          <Badge 
                            key={`importance-${index}-${imp}`} 
                            variant="outline" 
                            className="bg-muted text-foreground border-border"
                          >
                            Важность: {imp}
                          </Badge>
                        ))}
                      </>
                    );
                  })()}
                </div>
              </div>

              {/* Metadata */}
              {nodeDetails?.metadata && Object.keys(nodeDetails.metadata).length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-2 mt-6">Метаданные</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(nodeDetails.metadata).map(([key, value]: [string, any]) => (
                      <div key={key} className="bg-muted p-2 rounded-md">
                        <p className="text-xs text-muted-foreground">{key}</p>
                        <p className="text-sm font-medium">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Relations tab */}
          <TabsContent value="relations" className="overflow-y-auto p-0 h-full">
            <div className="p-4 pb-16">
              {/* Входящие связи */}
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Входящие связи</h3>
              <ul className="space-y-3 mb-6">
                {getIncomingNodes().length > 0 ? (
                  getIncomingNodes().map((related: {id: string; name: string; relevance: number; relationDescription?: string; relation_type?: string; direction?: string}, i: number) => (
                    <li key={i} className="flex items-center gap-2 justify-between border-b border-border pb-2">
                      <div className="flex items-center gap-3">
                        <div
                          className={cn(
                            "w-8 h-8 rounded-md flex items-center justify-center",
                            related.relevance > 80
                              ? "bg-green-500/20 text-green-600"
                              : related.relevance > 60
                              ? "bg-blue-500/20 text-blue-600"
                              : "bg-gray-500/20 text-gray-600",
                          )}
                        >
                          <span className="text-xs font-medium">{related.relevance}%</span>
                        </div>
                        <div>
                          <div className="text-sm font-medium">{related.name}</div>
                          {related.relationDescription && (
                            <div className="text-xs text-muted-foreground">
                              {related.relationDescription}
                              {related.relation_type && ` (${related.relation_type})`}
                            </div>
                          )}
                        </div>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-8"
                        onClick={() => handleNodeClick(related.id)}
                      >
                        Открыть
                        <ArrowRight className="ml-1 h-3 w-3" />
                      </Button>
                    </li>
                  ))
                ) : (
                  <li className="text-sm text-muted-foreground py-2">Нет входящих связей</li>
                )}
              </ul>
              
              {/* Исходящие связи */}
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Исходящие связи</h3>
              <ul className="space-y-3">
                {getOutgoingNodes().length > 0 ? (
                  getOutgoingNodes().map((related: {id: string; name: string; relevance: number; relationDescription?: string; relation_type?: string; direction?: string}, i: number) => (
                    <li key={i} className="flex items-center gap-2 justify-between border-b border-border pb-2">
                      <div className="flex items-center gap-3">
                        <div
                          className={cn(
                            "w-8 h-8 rounded-md flex items-center justify-center",
                            related.relevance > 80
                              ? "bg-green-500/20 text-green-600"
                              : related.relevance > 60
                              ? "bg-blue-500/20 text-blue-600"
                              : "bg-gray-500/20 text-gray-600",
                          )}
                        >
                          <span className="text-xs font-medium">{related.relevance}%</span>
                        </div>
                        <div>
                          <div className="text-sm font-medium">{related.name}</div>
                          {related.relationDescription && (
                            <div className="text-xs text-muted-foreground">
                              {related.relationDescription}
                              {related.relation_type && ` (${related.relation_type})`}
                            </div>
                          )}
                        </div>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-8"
                        onClick={() => handleNodeClick(related.id)}
                      >
                        Открыть
                        <ArrowRight className="ml-1 h-3 w-3" />
                      </Button>
                    </li>
                  ))
                ) : (
                  <li className="text-sm text-muted-foreground py-2">Нет исходящих связей</li>
                )}
              </ul>
            </div>
          </TabsContent>

          {/* Questions tab */}
          <TabsContent value="questions" className="overflow-y-auto p-0 h-full">
            <div className="p-4 pb-16">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">Проверьте свои знания</h3>
              
              {questionsLoading ? (
                <div className="flex flex-col items-center justify-center py-8">
                  <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
                  <p className="text-sm text-muted-foreground">Генерация вопросов...</p>
                </div>
              ) : (
                <ul className="space-y-4">
                  {getQuestions().map((question, i) => (
                    <li key={i} className="bg-muted rounded-lg p-3">
                      <p className="text-sm font-medium">{question}</p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </>
  )
}
