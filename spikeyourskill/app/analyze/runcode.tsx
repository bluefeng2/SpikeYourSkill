import { execFile } from 'child_process';
import type { NextApiRequest, NextApiResponse } from 'next';
import path from 'path';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { a } = req.query;

  if (!a) {
    return res.status(400).json({ error: 'Missing a or b' });
  }

  const scriptPath = path.resolve('./main.py');

  execFile('python3', [scriptPath, String(a)], (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: stderr || error.message });
    }

    res.status(200).json({ result: parseFloat(stdout.trim()) });
  });
}