"use client";

import { useState, useRef } from "react";

export default function FormPl({ onSubmitSuccess }: { onSubmitSuccess?: () => void }) {
  const [vid, setVid] = useState<File | null>(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!vid) {
      setError("Please select a video file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", vid);

    try {
      const res = await fetch("http://127.0.0.1:8000/runcode", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        setResult(data.duration);
        if (onSubmitSuccess) onSubmitSuccess();
      } else {
        setError(data.error || "Error occurred.");
      }
    } catch (err) {
      setError("Could not connect to API.");
    }
  };

  return (
    <main>
      <div className="flex flex-col items-center mb-4">
        <h1 className="font-ubuntu text-4xl font-bold text-blue-800">
          Input Video
        </h1>
      </div>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col text-center justify-center"
      >
        <div>
          {vid && <p className="mb-4">Selected Video: {vid.name}</p>}
          <input
            type="file"
            accept="video/*"
            ref={fileInputRef}
            onChange={(e) => setVid(e.target.files?.[0] || null)}
            required
            style={{ display: "none" }}
          />
          <button className="px-2">
            <label
              className="cursor-pointer bg-purple-800 text-white rounded-md px-4 py-2 hover:bg-blue-700 transition"
              onClick={() => fileInputRef.current?.click()}
            >
              Select Video
            </label>
          </button>

          <button type="submit" className="px-2">
            <label className="cursor-pointer bg-purple-800 text-white rounded-md px-4 py-2 hover:bg-blue-700 transition">
              Submit
            </label>
          </button>
        </div>
      </form>

      {result !== null && (
        <div className="justify-evenly flex gap-4 mt-4">
          <div className="rounded-xl bg-purple-100 p-4 min-w-[200px] text-center shadow">
            <p>Gemini Yap Divided Into Categories</p>
          </div>
          <div className="rounded-xl bg-purple-100 p-4 min-w-[200px] text-center shadow">
            <p>Image/Graph Row</p>
          </div>
        </div>
      )}
      {error && (
        <p className="text-center mt-4 font-ubuntu text-4xl font-bold text-red-800">
          {error}
        </p>
      )}
    </main>
  );
}