import { NextRequest, NextResponse } from 'next/server';
import { generatePlaceholderUrl, generateAsciiId, generateImagePaths, fetchCurrentGraph, initImageHelper } from '@/lib/node-image-helper';
import fs from 'fs';
import path from 'path';

// Добавляем глобальное объявление типа для NodeJS namespace
declare global {
  namespace NodeJS {
    interface Global {
      currentGraph?: string;
    }
  }
  
  // Для более новых версий Node.js
  var currentGraph: string | undefined;
}

// Функция для получения визуального placeholder для узла по его ID
export async function GET(request: NextRequest) {
  try {
    // Получаем ID узла и тип из URL параметров
    const nodeId = request.nextUrl.searchParams.get('id');
    const nodeType = request.nextUrl.searchParams.get('type') || 'concept';
    
    if (!nodeId) {
      // Если ID не предоставлен, возвращаем дефолтный placeholder
      return new NextResponse(null, {
        status: 302,
        headers: {
          'Location': generatePlaceholderUrl(nodeType)
        }
      });
    }
    
    console.log(`Searching image for node: ${nodeId}`);
    
    // Инициализируем и получаем директорию изображений для текущего графа
    const imagesDir = await initAsync();
    console.log(`Using images directory: ${imagesDir}`);
    
    // Генерируем несколько вариантов имен файлов для поиска
    const possibleImagePaths = generatePossibleImagePaths(nodeId);
    
    // Ищем первый существующий файл
    let foundImagePath = null;
    
    for (const filePath of possibleImagePaths) {
      const fullPath = path.join(process.cwd(), '..', imagesDir, filePath);
      console.log(`Checking path: ${fullPath}`);
      
      if (fs.existsSync(fullPath)) {
        foundImagePath = `/${imagesDir}/${filePath}`;
        console.log(`Found image: ${foundImagePath}`);
        break;
      }
    }
    
    // Если изображение не найдено в текущей директории, проверяем другие директории
    if (!foundImagePath) {
      // Проверяем в других директориях
      const directories = [
        'static/biology-images',
        'static/physics-images',
        'static/history-images',
        'static/images'
      ].filter(dir => dir !== imagesDir); // Исключаем текущую директорию, которую уже проверили
      
      for (const dir of directories) {
        for (const filePath of possibleImagePaths) {
          const fullPath = path.join(process.cwd(), '..', dir, filePath);
          console.log(`Fallback checking: ${fullPath}`);
          
          if (fs.existsSync(fullPath)) {
            foundImagePath = `/${dir}/${filePath}`;
            console.log(`Found image (fallback): ${foundImagePath}`);
            break;
          }
        }
        
        if (foundImagePath) break;
      }
    }
    
    // Если изображение не найдено, используем placeholder
    if (!foundImagePath) {
      console.log(`No image found for node: ${nodeId}, using placeholder`);
      const placeholderUrl = generatePlaceholderUrl(nodeType, nodeId);
      return new NextResponse(null, {
        status: 302,
        headers: {
          'Location': placeholderUrl,
          'Cache-Control': 'max-age=86400'
        }
      });
    }
    
    // Кодируем URL перед передачей в заголовок Location
    const encodedUrl = encodeURI(foundImagePath);
    console.log(`Redirecting to encoded URL: ${encodedUrl}`);
    
    // Перенаправляем на найденное изображение
    return new NextResponse(null, {
      status: 302,
      headers: {
        'Location': encodedUrl,
        'Cache-Control': 'max-age=86400'
      }
    });
  } catch (error) {
    console.error('Error in node-image API:', error);
    // При ошибке возвращаем дефолтный placeholder
    return new NextResponse(null, {
      status: 302,
      headers: {
        'Location': `/placeholder.svg?height=100&width=100&text=E&bg=ff0000&fg=ffffff`
      }
    });
  }
}

// Инициализация с ожиданием
async function initAsync() {
  // На сервере нет window, но мы можем сделать запрос к собственному API
  try {
    // Получаем текущий граф через прямой запрос к API
    const apiUrl = process.env.API_URL || 'http://localhost:8001';
    const response = await fetch(`${apiUrl}/api/graph/current`);
    const data = await response.json();
    console.log('Server-side fetch current graph:', data);
    
    // Устанавливаем глобальную переменную для функции generateImagePaths
    if (data && data.success && data.graph_name) {
      global.currentGraph = data.graph_name;
      console.log(`Server-side set currentGraph to: ${data.graph_name}`);
      
      // Получаем соответствующий путь к изображениям
      if (data.graph_name === 'biology_graph') {
        return 'static/biology-images';
      } else if (data.graph_name === 'physics_graph') {
        return 'static/physics-images';
      } else if (data.graph_name === 'history_graph') {
        return 'static/history-images';
      }
    }
  } catch (error) {
    console.error('Error fetching current graph on server:', error);
  }
  
  return 'static/images'; // Возвращаем путь по умолчанию
}

// Функция для генерации возможных путей к изображению на основе ID узла
function generatePossibleImagePaths(nodeId: string): string[] {
  if (!nodeId) return [];
  
  // Очищаем ID от префиксов для дальнейшей работы
  const cleanId = nodeId.replace(/^node_/, '').replace(/^rp_node_/, '');
  
  // Варианты ID для поиска - только оригинальные версии без ASCII-транслитерации
  const variants = [
    // Оригинальный ID с префиксом node_
    `node_${cleanId}`,
    
    // Оригинальный ID (как есть)
    nodeId,
    
    // Без префиксов
    cleanId,
    
    // Только ключевая часть (удаляем числовой суффикс)
    cleanId.replace(/_\d+$/, ''),
    
    // Только первое слово ключевой части
    cleanId.split('_')[0]
  ];
  
  // Добавляем расширения файлов
  const extensions = ['.jpg', '.jpeg', '.png', '.gif'];
  
  // Комбинируем все варианты с расширениями
  const results: string[] = [];
  
  for (const variant of variants) {
    if (!variant) continue;
    
    for (const ext of extensions) {
      results.push(`${variant}${ext}`);
    }
  }
  
  console.log(`Generated possible image paths for node ${nodeId}:`, results);
  
  return Array.from(new Set(results)); // Удаляем дубликаты
} 