import { NextRequest, NextResponse } from 'next/server';
import { generateImagePaths, generateAsciiId } from '@/lib/node-image-helper';

// Функция для отладки путей изображений
export async function GET(request: NextRequest) {
  try {
    // Получаем ID узла из URL параметра
    const nodeId = request.nextUrl.searchParams.get('id');
    
    if (!nodeId) {
      return NextResponse.json({ error: 'No node ID provided' });
    }
    
    // Генерируем ASCII-версию ID
    const asciiId = generateAsciiId(nodeId);
    
    // Получаем все возможные пути к изображению
    const paths = generateImagePaths(nodeId);
    
    // Возвращаем информацию о путях
    return NextResponse.json({
      nodeId,
      asciiId,
      baseNameWithoutPrefix: nodeId.replace(/^rp_node_|^node_/, ''),
      keyPart: nodeId.replace(/^rp_node_|^node_/, '').split('_')[0],
      asciiPaths: paths.filter(path => /^[\x00-\x7F]*$/.test(path)).slice(0, 5),
      totalPaths: paths.length,
      asciiPathsCount: paths.filter(path => /^[\x00-\x7F]*$/.test(path)).length,
      firstAsciiPath: paths.find(path => /^[\x00-\x7F]*$/.test(path)) || null
    });
  } catch (error) {
    console.error('Error in node-image debug API:', error);
    return NextResponse.json({ error: 'Error processing debug request' }, { status: 500 });
  }
} 