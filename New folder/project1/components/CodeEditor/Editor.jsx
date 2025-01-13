'use client';
import { useState } from 'react';
import Toolbar from './Toolbar';

export default function Editor() {
  const [code, setCode] = useState('');

  const handleCodeChange = (e) => {
    setCode(e.target.value);
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      <Toolbar code={code} />
      <textarea
        className="w-full h-96 p-4 font-mono text-sm focus:outline-none"
        value={code}
        onChange={handleCodeChange}
        placeholder="Enter your Python code here..."
      />
    </div>
  );
}