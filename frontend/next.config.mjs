/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
  // Полностью отключаем индикаторы разработки (включая иконку React DevTools)
  devIndicators: false,
  async rewrites() {
    // Определяем адрес API бэкенда из переменной окружения или используем порт 8000 по умолчанию
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    
    return [
      // Проксируем запросы к API на бэкенд
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      // Проксируем запросы к статическим файлам
      {
        source: '/static/:path*',
        destination: `${backendUrl}/static/:path*`,
      },
      // Доступ к файлам saved_graphs
      {
        source: '/saved_graphs/:path*',
        destination: `${backendUrl}/saved_graphs/:path*`,
      },
      {
        source: '/static/images/:path*',
        destination: '/api/static-images/:path*',
      },
    ];
  },
}

export default nextConfig
