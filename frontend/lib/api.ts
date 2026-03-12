// API client for communicating with the backend

// Base URL for API requests
const API_BASE_URL = '/api'; // Используем относительный путь для проксирования через Next.js

// Helper function for handling API responses
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || `API error: ${response.status}`);
  }
  return response.json();
};

// Graph API
export const graphApi = {
  // Get all graph data
  getGraphData: async () => {
    try {
      // Загружаем из API через прокси Next.js
      const response = await fetch(`${API_BASE_URL}/graph/data`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('API Response from getGraphData:', data);
      
      // Если edges приходят в другом формате, конвертируем их
      if (data.edges && Array.isArray(data.edges)) {
        console.log('Original edges structure:', data.edges[0]);
      }
      
      return data;
    } catch (error) {
      console.error('Error fetching graph data:', error);
      
      // При ошибке возвращаем пустые данные
      return {
        nodes: [],
        edges: []
      };
    }
  },

  // Get a specific node
  getNode: async (nodeId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/graph/node/${nodeId}`);
      return handleResponse(response);
    } catch (error) {
      console.error('Error fetching node:', error);
      throw new Error(`Failed to fetch node: ${error}`);
    }
  },

  // Create a new node
  createNode: async (nodeData: any) => {
    const response = await fetch(`${API_BASE_URL}/graph/node`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(nodeData),
    });
    return handleResponse(response);
  },

  // Update an existing node
  updateNode: async (nodeId: string, nodeData: any) => {
    const response = await fetch(`${API_BASE_URL}/graph/node/${nodeId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(nodeData),
    });
    return handleResponse(response);
  },

  // Delete a node
  deleteNode: async (nodeId: string) => {
    const response = await fetch(`${API_BASE_URL}/graph/node/${nodeId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`Failed to delete node: ${response.status}`);
    }
    return true;
  },

  // Search nodes
  searchNodes: async (query: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/graph/search?query=${encodeURIComponent(query)}`);
      return handleResponse(response);
    } catch (error) {
      console.error('Error searching nodes:', error);
      // Возвращаем пустой результат
      return { results: [] };
    }
  },

  // Generate questions for a node
  generateQuestions: async (nodeId: string) => {
    try {
      console.log(`API: Вызов generateQuestions для узла ${nodeId}`);
      const apiUrl = `${API_BASE_URL}/graph/generate_questions`;
      console.log('API URL:', apiUrl);
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ node_id: nodeId }),
      });
      
      console.log('API: Получен ответ со статусом', response.status);
      
      // Проверяем наличие ответа
      if (!response.ok) {
        console.error('API: Ошибка ответа', response.status);
        throw new Error(`Server responded with ${response.status}`);
      }
      
      // Разбираем ответ
      const result = await response.json();
      console.log('API: Разобранный ответ', result);
      
      // Проверяем структуру ответа
      if (result && result.questions && Array.isArray(result.questions)) {
        return result;
      } else if (result && result.success === true) {
        // Если response в другом формате, но есть поле success=true
        console.log('API: Получен успешный ответ, но с другой структурой', result);
        return {
          questions: result.questions || result.data || []
        };
      } else {
        console.warn('API: Неожиданный формат ответа', result);
        throw new Error('Unexpected response format');
      }
    } catch (error) {
      console.log('API: Ошибка запроса, использую моковые данные', error);
      
      // Возвращаем моковые данные для вопросов
      return {
        questions: [
          "Что такое " + nodeId + "?",
          "Какие основные характеристики " + nodeId + "?",
          "Как " + nodeId + " связано с другими концепциями?",
          "В чем важность изучения " + nodeId + "?",
          "Какие примеры использования " + nodeId + " в реальном мире?",
        ]
      };
    }
  },

  // Get node edges 
  getNodeEdges: async (nodeId: string, subject?: string) => {
    try {
      console.log(`API: Запрос связей для узла ${nodeId}${subject ? ', предмет: ' + subject : ''}`);
      
      // Формируем URL запроса с параметрами
      let url = `/api/node-edges?id=${encodeURIComponent(nodeId)}`;
      if (subject) {
        url += `&subject=${encodeURIComponent(subject)}`;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Ошибка получения связей: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(`API: Получены связи для узла ${nodeId}:`, data);
      return data;
    } catch (error) {
      console.error('Ошибка при получении связей:', error);
      // В случае ошибки возвращаем пустой массив
      return { edges: [] };
    }
  },

  // Получить контекст для буллета
  getBulletContext: async (nodeId: string, chunkId: string) => {
    try {
      console.log(`API: Запрос контекста для узла ${nodeId}, чанка ${chunkId}`);
      
      const response = await fetch(`${API_BASE_URL}/graph/bullet_context`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ node_id: nodeId, chunk_id: chunkId }),
      });
      
      if (!response.ok) {
        console.error(`API: Ошибка получения контекста: ${response.status}`);
        throw new Error(`Ошибка получения контекста: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(`API: Получен контекст для узла ${nodeId}, чанка ${chunkId}:`, data);
      
      return data;
    } catch (error) {
      console.error('Ошибка при получении контекста буллета:', error);
      return { 
        success: false, 
        context: `Не удалось загрузить контекст: ${error}` 
      };
    }
  },

  // Получить статистику по фильтрам (количество узлов)
  getFilterStats: async () => {
    try {
      // Сначала пробуем запросить данные через новый API endpoint
      let response = await fetch(`${API_BASE_URL}/graph/filter_stats`);
      
      // Если эндпоинт не найден (404), пробуем альтернативный путь
      if (response.status === 404) {
        console.log('Filter stats endpoint not found, trying alternative path');
        response = await fetch(`/api/graph/filter_stats`);
      }
      
      if (!response.ok) {
        console.error(`Failed to fetch filter statistics: ${response.status}`);
        throw new Error(`Failed to fetch filter statistics: ${response.status}`);
      }
      
      const data = await handleResponse(response);
      
      // Проверяем структуру ответа
      if (data && data.success && data.stats) {
        return data.stats;
      } else {
        console.error('Unexpected response format from filter_stats API:', data);
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error fetching filter statistics:', error);
      
      // При ошибке возвращаем моковые данные
      return {
        subjects: {
          "Математика": 42,
          "Физика": 32,
          "Биология": 28,
          "Информатика": 15
        },
        grades: {
          "5 класс": 35,
          "6 класс": 42,
          "7 класс": 38,
          "8 класс": 26
        },
        topics: {
          "Алгебра": 14,
          "Геометрия": 20,
          "Механика": 16,
          "Электричество": 18
        },
        subtopics: {
          "Уравнения": 8,
          "Тригонометрия": 12,
          "Вектор": 10,
          "Магнитное поле": 14
        }
      };
    }
  },
};

// Chat API
export const chatApi = {
  // Send a message to the AI and get a response
  sendMessage: async (message: string, topic?: string) => {
    const response = await fetch(`${API_BASE_URL}/chat/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        message,
        topic
      }),
    });
    return handleResponse(response);
  },
  
  // Generate a mind map based on a topic or query
  generateMindMap: async (query: string) => {
    const response = await fetch(`${API_BASE_URL}/chat/mindmap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
    return handleResponse(response);
  }
};

// Export the API client
export default {
  graph: graphApi,
  chat: chatApi
}; 