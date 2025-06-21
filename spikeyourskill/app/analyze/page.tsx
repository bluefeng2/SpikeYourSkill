'use client';

import { useState } from "react";
import FormPl from "./formpl";
import TransitionBackground from "../components/TransitionBackground";
export default function Home() {
  const [submit, setSubmit] = useState(false);
  const gradient = submit
  ? "bg-gradient-to-br from-green-400 to-white transition-colors duration-700"
  : "bg-gradient-to-br from-blue-400 to-white transition-colors duration-700";
  return (
  <main>
    <TransitionBackground gradient={gradient}>
      <FormPl onSubmitSuccess={() => setSubmit(true)}></FormPl>
    </TransitionBackground>
  </main>
  );
}