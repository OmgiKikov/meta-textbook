import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

// API-endpoint для получения связей узла из файла edges.json
export async function GET(request: NextRequest) {
  try {
    // Получаем ID узла из параметров запроса
    const nodeId = request.nextUrl.searchParams.get('id');
    // Получаем предмет из параметров запроса (если есть)
    const subject = request.nextUrl.searchParams.get('subject');
    
    if (!nodeId) {
      return NextResponse.json(
        { error: 'Не указан ID узла' },
        { status: 400 }
      );
    }
    
    console.log(`API: Запрос связей для узла с ID: ${nodeId}, предмет: ${subject || 'не указан'}`);
    
    // Базовый путь к файлам графа
    let basePath = path.join(process.cwd(), '..');
    
    // Определяем путь к файлам в зависимости от предмета
    let graphDirectory = 'saved_graphs'; // Путь по умолчанию
    
    if (subject) {
      // Нормализуем название предмета (переводим в нижний регистр для сравнения)
      const normalizedSubject = subject.toLowerCase().trim();
      console.log(`API: Нормализованный предмет: ${normalizedSubject}`);
      
      // Расширенное сопоставление названий предметов
      if (normalizedSubject.includes('биолог')) {
        graphDirectory = 'biology_graph';
        console.log(`API: Найдено соответствие для Биологии: ${graphDirectory}`);
      } 
      else if (normalizedSubject.includes('истор')) {
        graphDirectory = 'history_graph';
        console.log(`API: Найдено соответствие для Истории: ${graphDirectory}`);
      } 
      else if (normalizedSubject.includes('физик') || normalizedSubject.includes('физич')) {
        graphDirectory = 'physics_graph';
        console.log(`API: Найдено соответствие для Физики: ${graphDirectory}`);
      }
      // Если не нашли по нормализованной версии, пробуем прямое преобразование
      else {
        // Преобразуем название предмета в имя директории
        const subjectToDir: Record<string, string> = {
          'Биология': 'biology_graph',
          'биология': 'biology_graph',
          'История': 'history_graph',
          'история': 'history_graph',
          'История России': 'history_graph',
          'история россии': 'history_graph',
          'Физика': 'physics_graph',
          'физика': 'physics_graph'
        };
        
        if (subject in subjectToDir) {
          graphDirectory = subjectToDir[subject];
          console.log(`API: Использую директорию для точного соответствия предмета ${subject}: ${graphDirectory}`);
        }
      }
    }
    
    console.log(`API: Итоговая директория для предмета "${subject || 'не указан'}": ${graphDirectory}`);
    
    // Пути к файлам
    const edgesFilePath = path.join(basePath, graphDirectory, 'edges.json');
    const nodesFilePath = path.join(basePath, graphDirectory, 'nodes.json');
    
    // Проверяем существование файлов
    if (!fs.existsSync(edgesFilePath)) {
      console.error(`API: Файл не найден: ${edgesFilePath}`);
      return NextResponse.json(
        { error: `Файл связей не найден для ${graphDirectory}` },
        { status: 404 }
      );
    }
    
    if (!fs.existsSync(nodesFilePath)) {
      console.error(`API: Файл не найден: ${nodesFilePath}`);
      return NextResponse.json(
        { error: `Файл узлов не найден для ${graphDirectory}` },
        { status: 404 }
      );
    }
    
    // Читаем содержимое файлов
    const edgesContent = fs.readFileSync(edgesFilePath, 'utf-8');
    const nodesContent = fs.readFileSync(nodesFilePath, 'utf-8');
    
    const edges = JSON.parse(edgesContent);
    const nodes = JSON.parse(nodesContent);
    
    // Создаем словарь узлов для быстрого доступа по ID
    const nodesMap = new Map();
    nodes.forEach((node: any) => {
      nodesMap.set(node.id, node);
    });
    
    // Фильтруем связи для указанного узла
    const nodeEdges = edges.filter(
      (edge: any) => edge.source_id === nodeId || edge.target_id === nodeId
    );
    
    console.log(`API: Найдено связей для узла ${nodeId}: ${nodeEdges.length}`);
    
    // Преобразуем связи в формат для отображения
    const formattedEdges = nodeEdges.map((edge: any) => {
      // Определяем, какой узел связан с текущим
      const relatedNodeId = edge.source_id === nodeId ? edge.target_id : edge.source_id;
      
      // Получаем информацию о связанном узле
      const relatedNode = nodesMap.get(relatedNodeId);
      
      // Определяем направление связи
      const isOutgoing = edge.source_id === nodeId;
      
      return {
        id: edge.id,
        relatedNodeId: relatedNodeId,
        // Используем title узла или генерируем имя из ID
        relatedNodeName: relatedNode ? 
          (relatedNode.title || relatedNode.name || relatedNodeId.replace(/^node_/, '').replace(/^rp_node_/, '').replace(/_/g, ' ')) :
          relatedNodeId.replace(/^node_/, '').replace(/^rp_node_/, '').replace(/_/g, ' '),
        // Используем description связи
        description: edge.description || 'связан с',
        strength: edge.strength || 3,
        type: edge.relation_type || 'logical',
        direction: isOutgoing ? 'outgoing' : 'incoming',
        meta: edge.meta || []
      };
    });
    
    // Возвращаем результаты
    return NextResponse.json({ 
      nodeId,
      totalEdges: nodeEdges.length,
      edges: formattedEdges
    });
    
  } catch (error) {
    console.error('API: Ошибка при получении связей:', error);
    return NextResponse.json(
      { error: 'Ошибка при получении связей' },
      { status: 500 }
    );
  }
} 