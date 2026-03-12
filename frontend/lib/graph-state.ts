// Модуль для управления состоянием графа
// Этот файл хранит глобальное состояние графа для использования в разных компонентах

declare global {
  interface Window {
    currentGraph?: string;
    imagesPath?: string;
  }
}

// Сохраняем информацию о текущем графе
export function setCurrentGraph(graphName: string) {
  if (typeof window !== 'undefined') {
    console.log(`Setting current graph to: ${graphName}`);
    window.currentGraph = graphName;
    
    // Устанавливаем соответствующий путь к изображениям
    if (graphName === 'biology_graph') {
      window.imagesPath = '/static/biology-images';
    } else if (graphName === 'physics_graph') {
      window.imagesPath = '/static/physics-images';
    } else if (graphName === 'history_graph') {
      window.imagesPath = '/static/history-images';
    } else {
      window.imagesPath = '/static/images';
    }
    
    console.log(`Set images path to: ${window.imagesPath}`);
  }
}

// Получаем текущий граф
export function getCurrentGraph(): string {
  if (typeof window !== 'undefined') {
    return window.currentGraph || '';
  }
  return '';
}

// Получаем путь к изображениям для текущего графа
export function getImagesPath(): string {
  if (typeof window !== 'undefined') {
    return window.imagesPath || '/static/images';
  }
  return '/static/images';
}

// Инициализируем состояние графа из API
export async function initGraphState() {
  if (typeof window !== 'undefined') {
    try {
      console.log('Initializing graph state...');
      const response = await fetch('/api/graph/current');
      const data = await response.json();
      
      if (data && data.success && data.graph_name) {
        setCurrentGraph(data.graph_name);
        console.log(`Graph state initialized with: ${data.graph_name}`);
        return data.graph_name;
      }
    } catch (error) {
      console.error('Error initializing graph state:', error);
    }
  }
  return null;
}

// Автоматически инициализируем при загрузке модуля в браузере
if (typeof window !== 'undefined') {
  console.log('Graph state module loaded, initializing...');
  initGraphState();
} 