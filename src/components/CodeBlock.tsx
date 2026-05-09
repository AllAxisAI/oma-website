import { useState } from 'react';

type TokenType = 'keyword' | 'string' | 'comment' | 'decorator' | 'number' | 'plain';

interface Token {
  text: string;
  type: TokenType;
}

const TOKEN_COLORS: Record<TokenType, string> = {
  keyword:   'text-blue-400',
  string:    'text-emerald-400',
  comment:   'text-neutral-500 italic',
  decorator: 'text-yellow-400',
  number:    'text-orange-300',
  plain:     'text-neutral-200',
};

function tokenizePython(code: string): Token[] {
  const tokens: Token[] = [];
  // Priority order: comments, triple-quoted strings, single-quoted strings, decorators, numbers, keywords
  const regex = /(#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|@\w+|\b\d+\.?\d*\b|\b(?:def|class|return|import|from|if|else|elif|for|while|in|not|and|or|is|True|False|None|self|super|pass|raise|yield|with|as|lambda|async|await|try|except|finally|del|global|nonlocal|assert|break|continue)\b)/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(code)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: code.slice(lastIndex, match.index), type: 'plain' });
    }
    const text = match[0];
    let type: TokenType;
    if (text.startsWith('#'))         type = 'comment';
    else if (text.startsWith('"') || text.startsWith("'")) type = 'string';
    else if (text.startsWith('@'))    type = 'decorator';
    else if (/^\d/.test(text))        type = 'number';
    else                              type = 'keyword';
    tokens.push({ text, type });
    lastIndex = match.index + text.length;
  }
  if (lastIndex < code.length) {
    tokens.push({ text: code.slice(lastIndex), type: 'plain' });
  }
  return tokens;
}

interface CodeBlockProps {
  code: string;
  language?: 'python' | 'bash' | 'text';
  filename?: string;
}

export default function CodeBlock({ code, language = 'python', filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* fallback */ }
  };

  const tokens = language === 'python' ? tokenizePython(code) : null;

  return (
    <div className="rounded-2xl border border-white/10 overflow-hidden my-6">
      <div className="flex items-center justify-between bg-black/50 px-4 py-2.5 border-b border-white/10">
        <span className="text-xs font-mono text-neutral-500 uppercase tracking-widest">
          {filename ?? language}
        </span>
        <button
          onClick={handleCopy}
          className="text-xs text-neutral-500 hover:text-white transition-colors px-2 py-1 rounded-lg hover:bg-white/10"
        >
          {copied ? '✓ Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto bg-[rgba(0,0,0,0.25)] p-5">
        <code className="font-mono text-sm leading-7">
          {tokens
            ? tokens.map((token, i) => (
                <span key={i} className={TOKEN_COLORS[token.type]}>{token.text}</span>
              ))
            : <span className="text-neutral-200">{code}</span>
          }
        </code>
      </pre>
    </div>
  );
}
