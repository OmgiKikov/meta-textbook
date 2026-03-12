import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { headers } from 'next/headers';

export async function GET(
  request: NextRequest,
  context: { params: { path: string[] } }
) {
  try {
    // В Next.js 15+ для доступа к динамическим параметрам нужно использовать await
    const pathParam = await Promise.resolve(context.params.path);
    
    // Получаем путь к запрашиваемому изображению
    const imagePath = Array.isArray(pathParam) 
      ? pathParam.join('/') 
      : pathParam;
    
    // Формируем абсолютный путь к файлу в родительской директории
    const staticImagePath = path.join(process.cwd(), '..', 'static', 'images', imagePath);
    console.log(`Looking for image at: ${staticImagePath}`);
    
    // Проверяем существование файла
    if (!fs.existsSync(staticImagePath)) {
      console.error(`Image not found: ${staticImagePath}`);
      // Возвращаем 404, если файл не найден
      return new NextResponse(null, {
        status: 404,
        statusText: 'Not Found'
      });
    }
    
    // Читаем файл изображения
    const imageBuffer = fs.readFileSync(staticImagePath);
    
    // Определяем MIME-тип на основе расширения файла
    const extension = path.extname(staticImagePath).toLowerCase();
    let contentType = 'application/octet-stream';
    
    if (extension === '.jpg' || extension === '.jpeg') {
      contentType = 'image/jpeg';
    } else if (extension === '.png') {
      contentType = 'image/png';
    } else if (extension === '.gif') {
      contentType = 'image/gif';
    } else if (extension === '.svg') {
      contentType = 'image/svg+xml';
    }
    
    console.log(`Serving image ${imagePath} with content type: ${contentType}`);
    
    // Возвращаем содержимое файла с правильным Content-Type
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=86400'
      }
    });
  } catch (error) {
    console.error('Error serving static image:', error);
    return new NextResponse(null, {
      status: 500,
      statusText: 'Internal Server Error'
    });
  }
} 