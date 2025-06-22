'use client';

import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="w-full text-white px-4 py-3 flex items-center justify-between bg-transparent shadow-none" style={{ backgroundColor: "transparent" }}>
      <div className="font-raleway text-xl tracking-wide">
        <Link href="/">Spike Your Skill</Link>
      </div>
      <div className="flex gap-6 font-raleway ml-0.5">
        <Link
          href="/about"
          className="px-4 py-2 text-center font-raleway"
        >
          About
        </Link>
        <Link
          href="/analyze"
          className="px-4 py-2 text-center font-raleway"
        >
          Analyze
        </Link>
      </div>
    </nav>
  );
}