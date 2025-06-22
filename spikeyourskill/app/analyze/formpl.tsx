'use client';
import ReactMarkdown from 'react-markdown';
import { useState, useRef } from 'react';
import JSZip from 'jszip';
import { image } from 'framer-motion/client';


export default function formpl() {
  const [vid, setVid] = useState<File | null>(null);
  const [result, setResult] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [gemini, setGem] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return; // Prevent double submit
    setError('');
    setResult(null);
    setImageUrls([]);
    setLoading(true);


    if (!vid) {
      setError('Please select a video file.');
      setLoading(false);
      return;
    }


    const formData = new FormData();
    formData.append('video', vid);


    try {
      const res = await fetch('http://127.0.0.1:5000/main', {
        method: 'POST',
        body: formData,
      });


      if (res.ok) {
        const blob = await res.blob();
        const zip = await JSZip.loadAsync(blob);
        const urls: string[] = [];


        await Promise.all(
          Object.keys(zip.files).map(async (filename) => {
            const file = zip.files[filename];
            if (!file.dir && /\.(png)$/i.test(filename)) {
              const fileData = await file.async("blob");
              const url = URL.createObjectURL(fileData);
              urls.push(url);
            }
            else if (!file.dir && /\.(txt)$/i.test(filename)) {
              const fileData = await file.async("text");
              setGem(fileData);
            }
          })
        );
        setImageUrls(urls);
      } else {
        const data = await res.json();
        setError(data.error || 'Error occurred.');
      }
    } catch (err) {
      setError('Could not connect to API.');
    }
    setLoading(false);
  };


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVid(e.target.files[0]);
      setFileName(e.target.files[0].name);
    }
  };


  return (
    <div className="min-h-screen flex flex-col items-center pt-10">
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-5 w-full max-w-md items-center mb-10 text-center"
      >
        <h2 className="text-center text-blue-700 text-2xl font-bold m-0">Volleyball Serve Analyzer</h2>
          Select video file:
  <div className="flex items-center gap-3">
    <label
      htmlFor="video-upload"
      className="cursor-pointer p-2 rounded border border-gray-800 bg-gray-800 text-sm"
    >
      Choose File
    </label>
    <span className="text-gray-500 text-sm">
      {fileName ? `Selected: ${fileName}` : '[No file chosen]'}
    </span>
  </div>
  <input
    id="video-upload"
    type="file"
    accept="video/*"
    ref={fileInputRef}
    onChange={handleFileChange}
    className="hidden" // Hide the actual input
    disabled={loading}
  />
        <button
          type="submit"
          className={`py-3 w-full bg-gradient-to-r from-blue-600 to-green-400 text-white rounded font-semibold text-lg hover:from-blue-700 hover:to-green-500 transition ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Upload and Analyze'}
        </button>
        {loading && (
          <div className="w-full flex justify-center py-2">
          <img
            src="/volleyball-spin.png"
            alt="Loading"
            className="h-8"
            style={{ animation: 'slideX 1.5s infinite alternate' }}
          />
          <style jsx>{`
            @keyframes slideX {
              0% { transform: translateX(-120px); }
              100% { transform: translateX(120px); }
            }
          `}</style>
        </div>
        )}
        {error && (
          <div className="text-red-600 bg-red-50 border border-red-200 rounded p-2 text-center w-full">
            {error}
          </div>
        )}
      </form>
      {(gemini || imageUrls.length > 0) && (
      <div className="flex w-full max-w-6xl min-h-[60vh] border rounded-lg shadow-lg">
        {/* Gemini Markdown */}
        <div className="w-1/2 pr-6 overflow-y-auto max-h-[70vh]">
          {gemini && (
            <div className="max-w-xl mx-auto bg-gray-50 p-6 rounded-lg shadow text-blue-800">
              <ReactMarkdown>{gemini}</ReactMarkdown>
            </div>
          )}
        </div>
        {/* Images */}
        <div className="w-1/2 flex flex-col items-center gap-8 overflow-y-auto max-h-[70vh]">
          {imageUrls.map((url, idx) => (
            <img
              key={idx}
              src={url}
              alt={`Result ${idx}`}
              className="w-full max-w-lg rounded-lg shadow"
            />
          ))}
        </div>
      </div>
      )}
    </div>
  );
}




