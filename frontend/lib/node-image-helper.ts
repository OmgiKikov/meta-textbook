// Вспомогательные функции для работы с изображениями узлов

// Импортируем функции управления состоянием графа
import { getCurrentGraph, getImagesPath, initGraphState } from './graph-state';

// Объявляем глобальные переменные для хранения информации
declare global {
  interface Window {
    currentGraph?: string;
    nodeMetadata?: Record<string, any>;
  }
}

/**
 * Генерирует URL для получения изображения узла
 * @param nodeId - ID узла
 * @param nodeType - Тип узла (для placeholder)
 * @returns URL изображения
 */
export function getNodeImageUrl(nodeId: string, nodeType: string = 'concept'): string {
  if (!nodeId) {
    console.log("getNodeImageUrl: Empty nodeId, returning placeholder");
    return generatePlaceholderUrl(nodeType, nodeId);
  }
  
  console.log(`getNodeImageUrl: Creating URL for node ${nodeId} of type ${nodeType}`);
  
  // Возвращаем прямую ссылку на метод API, который генерирует визуальный placeholder
  const url = `/api/node-image?id=${encodeURIComponent(nodeId)}&type=${encodeURIComponent(nodeType)}`;
  console.log(`getNodeImageUrl: Generated URL: ${url}`);
  
  return url;
}

/**
 * Генерирует ASCII-совместимый идентификатор из ID узла
 * @param nodeId - ID узла
 * @returns ASCII-совместимый ID
 */
export function generateAsciiId(nodeId: string): string {
  if (!nodeId) return '';
  
  // Транслитерация кириллицы в латиницу
  const translitMap: Record<string, string> = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 
    'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
  };
  
  // Преобразуем в нижний регистр и заменяем символы
  const result = nodeId.toLowerCase().split('').map(char => {
    if (translitMap[char]) return translitMap[char];
    if (/[a-z0-9_\-]/.test(char)) return char;
    return '_';
  }).join('');
  
  return result;
}

/**
 * Генерирует предсказуемый цвет на основе строки
 * @param str - Входная строка
 * @returns HEX-код цвета
 */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Используем предопределенную палитру для разных типов узлов
  const predefinedColors: Record<string, string> = {
    'concept': '#3b82f6', // синий
    'term': '#10b981',    // зеленый
    'person': '#f59e0b',  // оранжевый
    'event': '#ef4444',   // красный
    'location': '#8b5cf6', // фиолетовый
    'organization': '#06b6d4' // голубой
  };
  
  const nodeType = str.toLowerCase();
  if (predefinedColors[nodeType]) {
    return predefinedColors[nodeType];
  }
  
  // Если тип узла не определен, генерируем цвет по хешу ID
  let color = '#';
  for (let i = 0; i < 3; i++) {
    // Смещаем биты, чтобы получить разные компоненты цвета
    const value = (hash >> (i * 8)) & 0xFF;
    // Гарантируем, что цвет не будет слишком темным или светлым
    const normalizedValue = Math.min(Math.max(value, 80), 220);
    color += normalizedValue.toString(16).padStart(2, '0');
  }
  
  return color;
}

/**
 * Генерирует URL для placeholder изображения
 * @param nodeType - Тип узла
 * @param nodeId - ID узла (используется для генерации стабильного цвета)
 * @returns URL для placeholder изображения
 */
export function generatePlaceholderUrl(nodeType: string, nodeId: string = ''): string {
  // Определяем цвет на основе типа узла или ID
  const bgColor = stringToColor(nodeType);
  const textColor = '#ffffff';
  
  // Используем первую букву типа узла (в верхнем регистре) как текст
  const displayText = nodeType.charAt(0).toUpperCase();
  
  // Если используется placeholder.svg
  return `/placeholder.svg?height=100&width=100&text=${displayText}&bg=${bgColor.replace('#', '')}&fg=${textColor.replace('#', '')}`;
}

/**
 * Генерирует список возможных имен файлов на основе ID узла
 * (Оставлено для обратной совместимости и отладки)
 */
export function generateImagePaths(nodeId: string): string[] {
  if (!nodeId) return [];
  
  // Используем текущий предмет для определения пути
  let imagesPath = "/static/images"; // По умолчанию
  
  // Получаем текущий граф из состояния
  const currentGraph = getCurrentGraph();
  
  console.log("Current graph for images:", currentGraph);
  
  if (currentGraph === "biology_graph") {
    imagesPath = "/static/biology-images";
  } else if (currentGraph === "physics_graph") {
    imagesPath = "/static/physics-images";
  } else if (currentGraph === "history_graph") {
    imagesPath = "/static/history-images";
  }
  
  console.log("Using images path:", imagesPath);
  
  // Генерируем разные варианты ID
  const variants = [
    // ASCII-версия ID для совместимости
    generateAsciiId(nodeId),
    // Только базовая часть ID (без префикса и суффикса)
    generateAsciiId(nodeId.replace(/^rp_node_|^node_/, '').split('_')[0]),
    // Оригинальный ID (rp_node_биология_0)
    nodeId,
    // Без префикса (биология_0)
    nodeId.replace(/^rp_node_|^node_/, ''),
    // Только ключевая часть (биология)
    nodeId.replace(/^rp_node_|^node_/, '').split('_')[0],
    // Очищенное имя
    nodeId.replace(/[^\w\d\-_]/g, '_').toLowerCase()
  ];
  
  // Генерируем пути, используя правильную директорию изображений
  const paths = variants.map(variant => `${imagesPath}/${variant}.jpg`);
  console.log("Generated image paths:", paths);
  return paths;
}

// Функция для получения текущего графа из API напрямую
export async function fetchCurrentGraph() {
  try {
    console.log("Fetching current graph information...");
    
    // Используем функцию из graph-state
    const graphName = await initGraphState();
    
    console.log("Fetched current graph:", graphName);
    return graphName;
  } catch (error) {
    console.error('Ошибка при получении текущего графа:', error);
    return null;
  }
}

// Добавляем функцию для немедленного вызова при импорте модуля
export function initImageHelper() {
  if (typeof window !== 'undefined') {
    console.log("Initializing image helper...");
    
    // Инициализируем состояние графа
    initGraphState().then(graph => {
      console.log("Initialized currentGraph in image helper:", graph);
    });
  }
}

// Инициализация при загрузке модуля
initImageHelper(); 