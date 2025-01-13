import './globals.css'

export const metadata = {
  title: 'Python Code Visualizer',
  description: 'Interactive Python code visualization and enhancement tool',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}