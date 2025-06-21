'use client';

import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-purple-900 w-full text-white px-4 py-3 flex items-center justify-between">
      <div className="font-raleway text-xl tracking-wide">
        <Link href="/">Spike Your Skill</Link>
      </div>
      <div className="flex gap-6 font-raleway">
        <Link href="/analyze" className="hover:underline hover:text-blue-200 transition">Analyze</Link>
        <Link href="/about" className="hover:underline hover:text-blue-200 transition">About</Link>
      </div>
    </nav>
  );
}