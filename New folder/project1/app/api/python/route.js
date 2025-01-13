import { NextResponse } from 'next/server';
import { runPythonCode } from '@/lib/pythonRunner';

export async function POST(request) {
  try {
    const { code } = await request.json();
    const result = await runPythonCode(code);
    return NextResponse.json({ result });
  } catch (error) {
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}