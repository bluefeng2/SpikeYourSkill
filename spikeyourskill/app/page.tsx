"use client";
import TransitionBackground from "./components/TransitionBackground";
import Image from "next/image";

export default function Home() {
  return (
    <TransitionBackground gradient="bg-gradient-to-br from-[#2d014d] to-black transition-colors duration-700">
      <title>Spike Your Skill</title>

      <div className="flex flex-col min-h-screen items-center justify-center">
        <header className="mb-8 flex flex-col items-center">
          <Image
            src="/spike-logo.svg" // Place your logo in /public as spike-logo.svg
            alt="Spike Your Skill Logo"
            width={96}
            height={96}
            className="mb-4"
          />
            <h1 className="font-ubuntu text-5xl font-bold text-white mb-4 mt-2">
              Spike Your Skill
            </h1>
          <p className="font-raleway text-lg text-gray-525 text-center max-w-xl">
            Elevate your volleyball game with AI-powered analytics, video feedback,
            and personalized training tools.
          </p>
        </header>
        <main className="flex flex-col gap-4 w-full max-w-md">
            <a
            href="/analyze"
            className="bg-purple-700 text-white rounded-lg px-6 py-3 text-center font-semibold hover:bg-purple-800 transition"
            >
            Analyze My Spike
            </a>
            <a
            href="/about"
            className="bg-white border border-[#7c3aed] text-blue-600 rounded-lg px-6 py-3 text-center font-semibold hover:bg-blue-50 transition"
            >
            Learn More
            </a>
        </main>
      </div>
    </TransitionBackground>
  );
}
