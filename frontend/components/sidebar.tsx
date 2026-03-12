"use client"

import * as React from "react"
import { ChevronRight, Hash, BookOpen, Layers, Compass, ChevronLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Checkbox } from "@/components/ui/checkbox"
import { ThemeToggle } from "@/components/ui/theme-toggle"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useToast } from "@/components/ui/use-toast"

// API клиент
import { graphApi } from "@/lib/api"

// Импортируем систему событий фильтрации
import { filterEvents } from "@/app/page"

interface FilterItemProps {
  label: string
  isSelected?: boolean
  onClick: () => void
  count?: number
}

export function FilterItem({ label, isSelected = false, onClick, count }: FilterItemProps) {
  return (
    <div className="flex items-center space-x-2 py-1.5 px-1">
      <Checkbox id={label} checked={isSelected} onCheckedChange={() => onClick()} />
      <label
        htmlFor={label}
        className={cn(
          "flex-1 text-sm cursor-pointer flex items-center justify-between",
          isSelected ? "text-foreground font-medium" : "text-foreground/70",
        )}
      >
        <span>{label}</span>
        {count !== undefined && (
          <Badge variant="outline" className="ml-auto text-xs font-normal">
            {count}
          </Badge>
        )}
      </label>
    </div>
  )
}

interface NavItemProps {
  icon: React.ReactNode
  label: string
  isSelected?: boolean
  onClick: () => void
  badge?: number
}

export function NavItem({ icon, label, isSelected = false, onClick, badge }: NavItemProps) {
  return (
    <div
      className={cn(
        "flex items-center py-2 px-3 rounded-md cursor-pointer transition-all duration-200",
        isSelected
          ? "bg-accent text-accent-foreground font-medium"
          : "hover:bg-accent/50 text-foreground/70 hover:text-foreground",
      )}
      onClick={onClick}
    >
      <span className="mr-2">{icon}</span>
      <span className="text-sm">{label}</span>
      {badge && (
        <Badge variant="outline" className="ml-auto">
          {badge}
        </Badge>
      )}
    </div>
  )
}

export function Sidebar({ onClose }: { onClose?: () => void }) {
  const [selectedNav, setSelectedNav] = React.useState("knowledge-graph")
  const { toast } = useToast()
  
  // Состояния для фильтров
  const [selectedSubjects, setSelectedSubjects] = React.useState<string[]>([])
  const [selectedGrades, setSelectedGrades] = React.useState<string[]>([])
  const [selectedTopics, setSelectedTopics] = React.useState<string[]>([])
  const [selectedSubtopics, setSelectedSubtopics] = React.useState<string[]>([])
  
  // Состояния для хранения опций фильтров из API
  const [subjects, setSubjects] = React.useState<string[]>([])
  const [grades, setGrades] = React.useState<string[]>([])
  const [topics, setTopics] = React.useState<string[]>([])
  const [subtopics, setSubtopics] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  
  // Добавляем состояние для текущего графа
  const [selectedGraph, setSelectedGraph] = React.useState<string>("")
  const [switchingGraph, setSwitchingGraph] = React.useState(false)

  // Загружаем текущий граф при инициализации
  React.useEffect(() => {
    const fetchCurrentGraph = async () => {
      try {
        const response = await fetch('/api/graph/current');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.graph_name) {
            console.log("Текущий активный граф:", data.graph_name);
            setSelectedGraph(data.graph_name);
          } else {
            console.error("Неверный формат ответа от API");
            setSelectedGraph("biology_graph"); // Используем значение по умолчанию
          }
        } else {
          console.error("Ошибка при получении текущего графа:", response.status);
          setSelectedGraph("biology_graph"); // Используем значение по умолчанию
        }
      } catch (error) {
        console.error("Не удалось получить текущий граф:", error);
        setSelectedGraph("biology_graph"); // Используем значение по умолчанию
      }
    };

    fetchCurrentGraph();
  }, []);

  // Функция для переключения графа
  const switchGraph = async (graphName: string) => {
    if (selectedGraph === graphName) return // Если тот же граф, не делаем ничего
    
    setSwitchingGraph(true)
    try {
      const response = await fetch('/api/graph/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ graph_name: graphName })
      })
      
      const data = await response.json()
      
      if (response.ok && data.success) {
        setSelectedGraph(graphName)
        toast({
          title: "Граф изменен",
          description: `Переключено на граф: ${getGraphDisplayName(graphName)}`,
        })
        
        // Сбрасываем выбранные фильтры
        resetAllFilters()
        
        // Перезагружаем страницу для обновления данных
        window.location.reload()
      } else {
        throw new Error(data.detail || "Ошибка при переключении графа")
      }
    } catch (error) {
      console.error("Ошибка при переключении графа:", error)
      toast({
        title: "Ошибка",
        description: `Не удалось переключить граф: ${error instanceof Error ? error.message : String(error)}`,
        variant: "destructive"
      })
    } finally {
      setSwitchingGraph(false)
    }
  }

  // Отображаем названия графов в более дружественном формате
  const getGraphDisplayName = (graphName: string) => {
    switch (graphName) {
      case "biology_graph": return "Биология 5 класс"
      case "physics_graph": return "Физика 10 класс"
      case "history_graph": return "История России 11 класс"
      default: return graphName
    }
  }

  // Загрузка опций метаданных при монтировании компонента
  React.useEffect(() => {
    const fetchMetadataOptions = async () => {
      setLoading(true)
      try {
        console.log("Выполняем запрос метаданных...");
        const response = await fetch('/api/graph/metadata_options');
        
        // Проверяем статус ответа
        if (!response.ok) {
          throw new Error(`Ошибка запроса: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json()
        
        console.log("Ответ API metadata_options:", data); // Логируем весь ответ
        
        if (data.success && data.options) {
          console.log("Полученные данные фильтров:", {
            subjects: data.options.subjects,
            grades: data.options.grades,
            topics: data.options.topics,
            subtopics: data.options.subtopics
          });
          
          // Проверяем есть ли данные и добавляем их
          setSubjects(data.options.subjects?.length ? data.options.subjects : ["Математика", "Физика", "Биология", "Информатика"]);
          setGrades(data.options.grades?.length ? data.options.grades : ["5 класс", "6 класс", "7 класс", "8 класс"]);
          setTopics(data.options.topics?.length ? data.options.topics : ["Алгебра", "Геометрия", "Механика", "Электричество"]);
          setSubtopics(data.options.subtopics?.length ? data.options.subtopics : ["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]);
        } else {
          console.error("Ошибка при загрузке опций метаданных:", data)
          
          // Используем моковые данные в случае ошибки
          setSubjects(["Математика", "Физика", "Биология", "Информатика"]);
          setGrades(["5 класс", "6 класс", "7 класс", "8 класс"]);
          setTopics(["Алгебра", "Геометрия", "Механика", "Электричество"]);
          setSubtopics(["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]);
          
          toast({
            title: "Ошибка",
            description: "Не удалось загрузить опции фильтров. Используются временные данные.",
            variant: "destructive"
          })
        }
      } catch (error) {
        console.error("Ошибка при загрузке опций метаданных:", error)
        
        // Используем моковые данные в случае ошибки
        setSubjects(["Математика", "Физика", "Биология", "Информатика"]);
        setGrades(["5 класс", "6 класс", "7 класс", "8 класс"]);
        setTopics(["Алгебра", "Геометрия", "Механика", "Электричество"]);
        setSubtopics(["Уравнения", "Тригонометрия", "Вектор", "Магнитное поле"]);
        
        toast({
          title: "Ошибка",
          description: "Не удалось загрузить опции фильтров: " + (error instanceof Error ? error.message : String(error)),
          variant: "destructive"
        })
      } finally {
        setLoading(false)
      }
    }

    fetchMetadataOptions()
  }, [toast])

  // Обработчики переключения фильтров
  const toggleSubject = (subject: string) => {
    const newSubjects = selectedSubjects.includes(subject) 
      ? selectedSubjects.filter(s => s !== subject) 
      : [...selectedSubjects, subject];
      
    setSelectedSubjects(newSubjects);
    notifyFilterChange(newSubjects, selectedGrades, selectedTopics, selectedSubtopics);
  }

  const toggleGrade = (grade: string) => {
    const newGrades = selectedGrades.includes(grade) 
      ? selectedGrades.filter(g => g !== grade) 
      : [...selectedGrades, grade];
      
    setSelectedGrades(newGrades);
    notifyFilterChange(selectedSubjects, newGrades, selectedTopics, selectedSubtopics);
  }

  const toggleTopic = (topic: string) => {
    const newTopics = selectedTopics.includes(topic) 
      ? selectedTopics.filter(t => t !== topic) 
      : [...selectedTopics, topic];
      
    setSelectedTopics(newTopics);
    notifyFilterChange(selectedSubjects, selectedGrades, newTopics, selectedSubtopics);
  }

  const toggleSubtopic = (subtopic: string) => {
    const newSubtopics = selectedSubtopics.includes(subtopic) 
      ? selectedSubtopics.filter(s => s !== subtopic) 
      : [...selectedSubtopics, subtopic];
      
    setSelectedSubtopics(newSubtopics);
    notifyFilterChange(selectedSubjects, selectedGrades, selectedTopics, newSubtopics);
  }

  // Функция для уведомления о изменении фильтров
  const notifyFilterChange = (
    subjects: string[] = selectedSubjects,
    grades: string[] = selectedGrades,
    topics: string[] = selectedTopics,
    subtopics: string[] = selectedSubtopics
  ) => {
    // Уведомляем всех подписчиков о изменении фильтров
    filterEvents.notify({
      subjects,
      grades,
      topics,
      subtopics
    });
  }

  // Функция для сброса всех фильтров
  const resetAllFilters = () => {
    setSelectedSubjects([])
    setSelectedGrades([])
    setSelectedTopics([])
    setSelectedSubtopics([])
    
    // Уведомляем о сбросе фильтров
    notifyFilterChange([], [], [], [])
  }

  return (
    <div className="w-[280px] bg-background border-r border-border flex flex-col h-screen relative">
      {/* Кнопка закрытия */}
      {onClose && (
        <button
          className="absolute top-4 right-4 z-20 bg-background border border-border rounded-full shadow p-1 hover:bg-accent transition"
          onClick={onClose}
          aria-label="Скрыть меню"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
      )}
      <div className="p-4 flex items-center justify-between border-b border-border">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-md bg-foreground flex items-center justify-center mr-3">
            <span className="text-sm font-medium text-background">М</span>
          </div>
          <span className="text-base font-medium">Meta-Textbook</span>
        </div>
        <ThemeToggle />
      </div>

      <div className="flex-1 overflow-y-auto">
        <Separator className="my-2" />

        <div className="p-3">
          <div className="flex items-center justify-between mb-1 px-2">
            <span className="text-xs font-medium text-muted-foreground">ФИЛЬТРЫ</span>
            <Button 
              variant="ghost" 
              size="sm" 
              className="h-6 text-xs px-2 text-muted-foreground hover:text-foreground"
              onClick={resetAllFilters}
            >
              Сбросить
            </Button>
          </div>

          {loading ? (
            <div className="p-4 text-center">
              <div className="animate-spin w-4 h-4 border-2 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-xs text-muted-foreground">Загрузка фильтров...</p>
            </div>
          ) : (
            <>
              {/* Секция выбора графа */}
              <div className="mb-6 px-2">
                <h3 className="text-sm font-medium mb-3">Выберите материал</h3>
                <div className="space-y-2">
                  {!selectedGraph ? (
                    // Показываем индикатор загрузки, пока текущий граф не определен
                    <div className="py-4 text-center">
                      <div className="animate-spin w-4 h-4 border-2 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
                      <p className="text-xs text-muted-foreground">Загрузка...</p>
                    </div>
                  ) : (
                    <>
                      <Button
                        variant={selectedGraph === "biology_graph" ? "default" : "outline"}
                        className="w-full justify-start"
                        onClick={() => switchGraph("biology_graph")}
                        disabled={switchingGraph}
                      >
                        <BookOpen className="mr-2 h-4 w-4" />
                        Биология 5 класс
                        {switchingGraph && selectedGraph === "biology_graph" && <span className="ml-2 animate-spin">⟳</span>}
                      </Button>
                      
                      <Button
                        variant={selectedGraph === "physics_graph" ? "default" : "outline"}
                        className="w-full justify-start"
                        onClick={() => switchGraph("physics_graph")}
                        disabled={switchingGraph}
                      >
                        <BookOpen className="mr-2 h-4 w-4" />
                        Физика 10 класс
                        {switchingGraph && selectedGraph === "physics_graph" && <span className="ml-2 animate-spin">⟳</span>}
                      </Button>
                      
                      <Button
                        variant={selectedGraph === "history_graph" ? "default" : "outline"}
                        className="w-full justify-start"
                        onClick={() => switchGraph("history_graph")}
                        disabled={switchingGraph}
                      >
                        <BookOpen className="mr-2 h-4 w-4" />
                        История России 11 класс
                        {switchingGraph && selectedGraph === "history_graph" && <span className="ml-2 animate-spin">⟳</span>}
                      </Button>
                    </>
                  )}
                </div>
              </div>
            
              <Accordion type="multiple" defaultValue={["тема", "подтема"]} className="w-full">
                {/* Убираем фильтры предметов и классов, оставляем только темы и подтемы */}
                
                {/* Фильтр Темы */}
                <AccordionItem value="тема" className="border-none">
                  <AccordionTrigger className="py-1.5 px-2 text-sm hover:no-underline">
                    <span className="text-sm font-medium">Тема</span>
                  </AccordionTrigger>
                  <AccordionContent className="pl-2">
                    <div className="max-h-[180px] overflow-y-auto pr-1 scrollbar-thin">
                      {topics.length > 0 ? topics.map((topic) => (
                        <FilterItem
                          key={topic}
                          label={topic}
                          isSelected={selectedTopics.includes(topic)}
                          onClick={() => toggleTopic(topic)}
                        />
                      )) : (
                        <div className="py-1 px-2 text-xs text-muted-foreground">Нет доступных тем</div>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>

                {/* Фильтр Подтемы */}
                <AccordionItem value="подтема" className="border-none">
                  <AccordionTrigger className="py-1.5 px-2 text-sm hover:no-underline">
                    <span className="text-sm font-medium">Подтема</span>
                  </AccordionTrigger>
                  <AccordionContent className="pl-2">
                    <div className="max-h-[180px] overflow-y-auto pr-1 scrollbar-thin">
                      {subtopics.length > 0 ? subtopics.map((subtopic) => (
                        <FilterItem
                          key={subtopic}
                          label={subtopic}
                          isSelected={selectedSubtopics.includes(subtopic)}
                          onClick={() => toggleSubtopic(subtopic)}
                        />
                      )) : (
                        <div className="py-1 px-2 text-xs text-muted-foreground">Нет доступных подтем</div>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </>
          )}
        </div>
      </div>

      <div className="p-3 border-t border-border">
        {/* Documentation button removed */}
      </div>
    </div>
  )
}
