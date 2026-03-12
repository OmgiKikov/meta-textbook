"use client"

import { useState, useEffect } from "react"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"
import { NodeBadge } from "@/components/node-badge"
import { getNodeImageUrl } from "@/lib/node-image-helper"

interface NodePreviewProps {
  node: any
  position: { x: number; y: number }
  onClose: () => void
  onViewDetails: () => void
  onMouseEnter?: () => void
  onMouseLeave?: () => void
  className?: string
}

export function NodePreview({
  node,
  position,
  onClose,
  onViewDetails,
  onMouseEnter,
  onMouseLeave,
  className,
}: NodePreviewProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [imageError, setImageError] = useState(false)
  const router = useRouter()
  
  console.log("Node preview data:", node); // Отладочный вывод для проверки данных

  // Animate in
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true)
    }, 50)
    return () => clearTimeout(timer)
  }, [])

  // При изменении узла сбрасываем ошибку изображения
  useEffect(() => {
    setImageError(false)
  }, [node])

  // Получаем название узла
  const getNodeTitle = () => {
    // Приоритет: оригинальные данные, label, title, name, id
    if (node.originalData?.title) return node.originalData.title;
    if (node.label) return node.label;
    if (node.title) return node.title;
    if (node.name) return node.name;
    return node.id || "Неизвестный узел";
  }

  // Вычисляем тип узла
  const getNodeType = () => {
    // Приоритет: оригинальные данные, type node, group
    if (node.originalData?.type) return node.originalData.type;
    if (node.type) return node.type;
    
    // По группе определяем тип
    const groupTypes = ["concept", "person", "event", "location", "organization", "term"];
    if (typeof node.group === "number" && node.group >= 0 && node.group < groupTypes.length) {
      return groupTypes[node.group];
    }
    
    return "concept";
  }

  // Используем description из узла или запасное значение
  const getNodeDefinition = () => {
    if (node.originalData?.description) return node.originalData.description;
    if (node.description) return node.description;
    return "Описание отсутствует";
  }

  // Получаем обрезанный summary для контекста
  const getNodeSummary = () => {
    let summary = "";
    
    if (node.originalData?.summary) {
      summary = node.originalData.summary;
    } else if (node.summary) {
      summary = node.summary;
    }
    
    if (!summary) return "";
    
    // Берем первый параграф из summary, разделенного маркерами ---
    const firstParagraph = summary.split('---')[0];
    
    // Обрезаем до 150 символов
    if (firstParagraph.length > 150) {
      return firstParagraph.substring(0, 147) + '...';
    }
    
    return firstParagraph;
  }

  // Получаем важность узла
  const getNodeImportance = () => {
    if (node.originalData?.importance) return node.originalData.importance;
    if (node.importance) return node.importance;
    if (node.val) return node.val;
    return 1;
  }

  // Получаем изображение на основе ID узла
  const getNodeImage = () => {
    // Если была ошибка загрузки, сразу возвращаем placeholder
    if (imageError) {
      const nodeType = getNodeType();
      return `/placeholder.svg?height=100&width=100&text=${nodeType}`;
    }
    
    // Получаем ID узла
    const nodeId = node.id || node.originalData?.id || "";
    const nodeType = getNodeType();
    
    // Логируем информацию о запросе изображения
    console.log("NodePreview: Requesting image for node:", {
      id: nodeId,
      type: nodeType,
      title: getNodeTitle()
    });
    
    // Используем наш хелпер для получения URL изображения
    const imageUrl = getNodeImageUrl(nodeId, nodeType);
    console.log("NodePreview: Generated image URL:", imageUrl);
    
    return imageUrl;
  }

  // Обработчик ошибки загрузки изображения
  const handleImageError = () => {
    console.log("Image load error, using placeholder");
    setImageError(true);
  }

  const handleViewDetails = () => {
    onViewDetails();
    // In a real app, we would navigate to a dedicated page
    // router.push(`/node/${node.id}`);
  }

  // Calculate position to ensure preview stays within viewport
  const calculatePosition = () => {
    const margin = 10;
    const previewWidth = 300;
    const previewHeight = 350;

    let x = position.x + 20;
    let y = position.y - 20;

    // Check right edge
    if (x + previewWidth > window.innerWidth - margin) {
      x = position.x - previewWidth - 20;
    }

    // Check bottom edge
    if (y + previewHeight > window.innerHeight - margin) {
      y = window.innerHeight - previewHeight - margin;
    }

    // Check top edge
    if (y < margin) {
      y = margin;
    }

    return { x, y };
  }

  const { x, y } = calculatePosition();

  if (!node) return null;

  // Получаем URL изображения
  const imageUrl = getNodeImage();
  // Получаем URL для placeholder
  const placeholderUrl = `/placeholder.svg?height=100&width=100&text=${getNodeType()}`;
  
  return (
    <Card
      className={cn(
        "absolute w-[300px] shadow-lg z-50 transform transition-all duration-200 ease-out",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2",
        className,
      )}
      style={{
        left: `${x}px`,
        top: `${y}px`,
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="relative overflow-hidden h-[100px] bg-muted">
        {!imageError ? (
          <img
            src={imageUrl}
            alt={getNodeTitle()}
            className="w-full h-full object-cover"
            onError={handleImageError}
          />
        ) : (
          // Отображаем placeholder если есть ошибка загрузки
          <div className="w-full h-full flex items-center justify-center bg-primary/10">
            <div className="text-center">
              <div className="text-4xl mb-1">{getNodeType()[0].toUpperCase()}</div>
              <div className="text-xs text-muted-foreground">{getNodeType()}</div>
            </div>
          </div>
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent flex items-end">
          <div className="p-3 w-full">
            <div className="flex items-center justify-between mb-1">
              <NodeBadge type={getNodeType()} />
            </div>
            <h3 className="text-white font-medium text-lg">{getNodeTitle()}</h3>
          </div>
        </div>
      </div>

      <CardContent className="p-4 flex flex-col h-[300px]">
        <div className="mb-4">
          <h4 className="text-sm font-medium text-muted-foreground mb-1">Группы:</h4>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="bg-primary/5">
              {getNodeType()}
            </Badge>
            <Badge variant="secondary">
              Важность: {getNodeImportance()}
            </Badge>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 pr-2 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">Определение:</h4>
            <p className="text-sm">{getNodeDefinition()}</p>
          </div>

          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">Контекст:</h4>
            <p className="text-sm">{getNodeSummary()}</p>
          </div>
        </div>
      </CardContent>

      <CardFooter className="border-t p-3 bg-muted/50">
        <Button onClick={handleViewDetails} className="w-full" variant="default">
          Подробнее
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}
