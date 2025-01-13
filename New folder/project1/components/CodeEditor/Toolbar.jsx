'use client';
import Button from '../UI/Button';
import { useState } from 'react';

export default function Toolbar({ code }) {
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = async () => {
    setIsRunning(true);
    try {
      const response = await fetch('/api/python', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const data = await response.json();
      // Handle the response
    } catch (error) {
      console.error('Error:', error);
    }
    setIsRunning(false);
  };

  return (
    <div className="bg-gray-100 p-2 flex space-x-2">
      <Button onClick={handleRun} disabled={isRunning}>
        {isRunning ? 'Running...' : 'Run'}
      </Button>
      <Button>Clear</Button>
    </div>
  );
}