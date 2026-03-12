import { useReducer } from "react";
import { type Message } from "ai/react";

// Типы для контекста mind map
export type MindMapContext = 
  | { type: "full" }
  | { type: "node"; nodeName: string; nodeContext?: string };

// Состояние компонента
export interface ChatState {
  currentContext: MindMapContext;
  currentMindMap: string | null;
  hasMindMap: boolean;
  isGeneratingMindMap: boolean;
  error: string | null;
  errorDetails: string | null;
  expansionSuggestions: string[];
  complexityLevel: string | undefined;
}

// Начальное состояние
export const initialState: ChatState = {
  currentContext: { type: "full" },
  currentMindMap: null,
  hasMindMap: false,
  isGeneratingMindMap: false,
  error: null,
  errorDetails: null,
  expansionSuggestions: [],
  complexityLevel: undefined,
};

// Типы действий для редьюсера
export type ChatAction =
  | { type: "SET_CONTEXT"; payload: MindMapContext }
  | { type: "SET_MINDMAP"; payload: string | null }
  | { type: "SET_HAS_MINDMAP"; payload: boolean }
  | { type: "SET_GENERATING"; payload: boolean }
  | { type: "SET_ERROR"; payload: { error: string | null; details?: string | null } }
  | { type: "SET_SUGGESTIONS"; payload: string[] }
  | { type: "SET_COMPLEXITY"; payload: string | undefined }
  | { type: "RESET_FOR_NEW_TOPIC" };

// Редьюсер для управления состоянием
export const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case "SET_CONTEXT":
      return { ...state, currentContext: action.payload };
    case "SET_MINDMAP":
      return { ...state, currentMindMap: action.payload };
    case "SET_HAS_MINDMAP":
      return { ...state, hasMindMap: action.payload };
    case "SET_GENERATING":
      return { ...state, isGeneratingMindMap: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload.error, errorDetails: action.payload.details || null };
    case "SET_SUGGESTIONS":
      return { ...state, expansionSuggestions: action.payload };
    case "SET_COMPLEXITY":
      return { ...state, complexityLevel: action.payload };
    case "RESET_FOR_NEW_TOPIC":
      return { ...initialState };
    default:
      return state;
  }
};

// Функция для извлечения контекста узла из mind map
export const extractNodeContext = (markdown: string, nodeName: string): string | null => {
  if (!markdown || !nodeName) return null;
  
  try {
    const lines = markdown.split('\n');
    const nodeNameLower = nodeName.toLowerCase().trim();
    
    // Найдем строку с названием узла (обычно он начинается с ##)
    const nodeLineIndex = lines.findIndex(line => {
      const lineContent = line.replace(/^#+\s+/, '').toLowerCase().trim();
      return lineContent === nodeNameLower || 
             lineContent.startsWith(nodeNameLower + ' ') ||
             lineContent.endsWith(' ' + nodeNameLower) ||
             lineContent.includes(' ' + nodeNameLower + ' ');
    });
    
    if (nodeLineIndex === -1) return null;
    
    // Определим уровень этого узла (количество #)
    const nodeLevel = (lines[nodeLineIndex].match(/^(#+)/) || ['#'])[0].length;
    
    // Найдем строки, относящиеся к этому узлу и его подузлам
    // Собираем все до следующего узла того же или более высокого уровня
    const relevantLines = [lines[nodeLineIndex]];
    
    // Извлекаем основную тему (# заголовок первого уровня)
    const mainTopic = lines.find(line => line.startsWith('# ')) || '# Тема';
    if (!relevantLines.includes(mainTopic)) {
      relevantLines.unshift(mainTopic);  // Добавляем главную тему в начало
    }
    
    // Собираем подузлы
    for (let i = nodeLineIndex + 1; i < lines.length; i++) {
      const line = lines[i];
      const match = line.match(/^(#+)/);
      
      // Если это не заголовок или уровень глубже, добавляем
      if (!match || match[0].length > nodeLevel) {
        relevantLines.push(line);
      } else {
        // Если встретили заголовок того же или более высокого уровня, останавливаемся
        break;
      }
    }
    
    return relevantLines.join('\n');
  } catch (error) {
    console.error("Error extracting node context:", error);
    return null;
  }
};

// Функция для генерации предложений по расширению на основе текущей mind map
export const generateExpansionSuggestions = (markdown: string): string[] => {
  if (!markdown || markdown.length < 10) return [];

  try {
    // Извлекаем заголовки разных уровней для предложений по расширению
    const lines = markdown.split('\n');
    
    // Извлекаем главную тему (# уровень)
    const mainTopic = lines
      .find(line => line.trim().startsWith('# '))
      ?.trim().replace('# ', '') || '';
    
    // Извлекаем заголовки второго уровня (## уровень)
    const secondLevelHeaders = lines
      .filter(line => line.trim().startsWith('## '))
      .map(line => line.trim().replace('## ', ''))
      .slice(0, 3); // Берем максимум 3 предложения
    
    // Создаем предложения на основе заголовков и темы
    const suggestions = [
      ...secondLevelHeaders.map(header => `Расскажи подробнее о "${header}"`),
    ];
    
    // Добавляем предложения на основе главной темы
    if (mainTopic) {
      suggestions.push(`Приведи примеры по теме "${mainTopic}"`);
      suggestions.push(`Объясни "${mainTopic}" простыми словами`);
      
      // Добавляем тематические предложения в зависимости от темы
      if (/атом|молекул|вещество|электрон|протон|химия/i.test(mainTopic)) {
        suggestions.push(`Расскажи о применении ${mainTopic} в науке`);
      } else if (/белок|днк|рнк|ген|клетка|биология/i.test(mainTopic)) {
        suggestions.push(`Какое значение имеют ${mainTopic} для жизни?`);
      } else if (/математика|формула|уравнение|функция/i.test(mainTopic)) {
        suggestions.push(`Приведи примеры задач с ${mainTopic}`);
      }
    }
    
    return suggestions;
  } catch (error) {
    console.error("Error generating expansion suggestions:", error);
    // Добавляем дефолтные предложения при ошибке
    return [
      'Добавь практические примеры',
      'Объясни простыми словами',
      'Расскажи подробнее об этой теме'
    ];
  }
};

// Функция для проверки изменения темы
export const isNewTopic = (newMessage: string, mindMapMarkdown: string | null): boolean => {
  if (!mindMapMarkdown) return true;
  
  // Извлекаем название основной темы из markdown (первая строка с # )
  const mainTopicMatch = mindMapMarkdown.match(/^#\s+(.+)$/m);
  const mainTopic = mainTopicMatch ? mainTopicMatch[1].toLowerCase() : '';
  
  // Проверяем, содержится ли текущая тема в новом сообщении
  // Если текущая тема отсутствует в новом сообщении, 
  // вероятно, это новая тема
  if (mainTopic && newMessage.toLowerCase().includes(mainTopic)) {
    return false; // Тема не изменилась
  }
  
  // Ищем ключевые слова "расскажи" или "объясни", которые могут указывать на новую тему
  const newTopicIndicators = /расскажи|объясни|что такое|как работает|что значит/i;
  return newTopicIndicators.test(newMessage);
}; 