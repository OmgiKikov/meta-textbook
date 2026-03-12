// Sample data for graph visualization
export const graphData = {
  nodes: Array.from({ length: 50 }, (_, i) => ({
    id: `node${i}`,
    group: Math.floor(Math.random() * 6),
    val: Math.random() * 20 + 5,
    label: `Концепт ${i + 1}`,
  })),
  links: Array.from({ length: 100 }, () => {
    const source = Math.floor(Math.random() * 50)
    let target = Math.floor(Math.random() * 50)
    // Ensure source and target are different
    while (source === target) {
      target = Math.floor(Math.random() * 50)
    }
    return {
      source: `node${source}`,
      target: `node${target}`,
      value: Math.random() * 2 + 0.5,
    }
  }),
}

// Sample data for mindmap visualization
export const mindmapData = {
  id: "root",
  name: "Математика",
  children: [
    {
      id: "algebra",
      name: "Алгебра",
      children: [
        {
          id: "equations",
          name: "Уравнения",
          children: [
            { id: "linear", name: "Линейные уравнения" },
            { id: "quadratic", name: "Квадратные уравнения" },
            { id: "cubic", name: "Кубические уравнения" },
          ],
        },
        {
          id: "functions",
          name: "Функции",
          children: [
            { id: "linear-func", name: "Линейные функции" },
            { id: "quadratic-func", name: "Квадратичные функции" },
            { id: "exponential", name: "Экспоненциальные функции" },
          ],
        },
      ],
    },
    {
      id: "geometry",
      name: "Геометрия",
      children: [
        {
          id: "plane",
          name: "Планиметрия",
          children: [
            { id: "triangles", name: "Треугольники" },
            { id: "circles", name: "Окружности" },
            { id: "polygons", name: "Многоугольники" },
          ],
        },
        {
          id: "solid",
          name: "Стереометрия",
          children: [
            { id: "polyhedra", name: "Многогранники" },
            { id: "spheres", name: "Сферы" },
            { id: "cylinders", name: "Цилиндры" },
          ],
        },
      ],
    },
    {
      id: "calculus",
      name: "Математический анализ",
      children: [
        { id: "limits", name: "Пределы" },
        { id: "derivatives", name: "Производные" },
        { id: "integrals", name: "Интегралы" },
      ],
    },
  ],
}

// Sample data for protein mind map
export function generateProteinMindMap() {
  return {
    id: "protein",
    name: "Белок",
    children: [
      {
        id: "structure",
        name: "Структура белка",
        children: [
          { id: "primary", name: "Первичная структура" },
          { id: "secondary", name: "Вторичная структура" },
          { id: "tertiary", name: "Третичная структура" },
          { id: "quaternary", name: "Четвертичная структура" },
        ],
      },
      {
        id: "functions",
        name: "Функции белков",
        children: [
          { id: "enzymatic", name: "Ферментативная" },
          { id: "transport", name: "Транспортная" },
          { id: "structural", name: "Структурная" },
          { id: "defense", name: "Защитная" },
          { id: "regulatory", name: "Регуляторная" },
        ],
      },
      {
        id: "synthesis",
        name: "Синтез белка",
        children: [
          { id: "transcription", name: "Транскрипция" },
          { id: "translation", name: "Трансляция" },
          { id: "folding", name: "Фолдинг" },
        ],
      },
      {
        id: "types",
        name: "Типы белков",
        children: [
          { id: "globular", name: "Глобулярные" },
          { id: "fibrous", name: "Фибриллярные" },
          { id: "membrane", name: "Мембранные" },
        ],
      },
    ],
  }
}

// Sample data for hierarchical subjects
export const subjectsData = [
  {
    id: "math",
    name: "Математика",
    icon: "🔢",
    topics: [
      { id: "algebra", name: "Алгебра", count: 24 },
      { id: "geometry", name: "Геометрия", count: 18 },
      { id: "calculus", name: "Математический анализ", count: 12 },
    ],
  },
  {
    id: "physics",
    name: "Физика",
    icon: "⚛️",
    topics: [
      { id: "mechanics", name: "Механика", count: 16 },
      { id: "electricity", name: "Электричество", count: 14 },
      { id: "optics", name: "Оптика", count: 8 },
    ],
  },
  {
    id: "chemistry",
    name: "Химия",
    icon: "🧪",
    topics: [
      { id: "organic", name: "Органическая химия", count: 22 },
      { id: "inorganic", name: "Неорганическая химия", count: 18 },
      { id: "biochemistry", name: "Биохимия", count: 10 },
    ],
  },
  {
    id: "biology",
    name: "Биология",
    icon: "🌱",
    topics: [
      { id: "botany", name: "Ботаника", count: 15 },
      { id: "zoology", name: "Зоология", count: 20 },
      { id: "genetics", name: "Генетика", count: 12 },
    ],
  },
  {
    id: "history",
    name: "История",
    icon: "🏛️",
    topics: [
      { id: "ancient", name: "Древний мир", count: 18 },
      { id: "medieval", name: "Средние века", count: 14 },
      { id: "modern", name: "Новое время", count: 22 },
    ],
  },
]
