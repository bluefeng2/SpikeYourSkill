'use client';
import { motion } from "framer-motion";
import { ReactNode } from "react";

export default function TransitionBackground({
  gradient,
  children,
}: {
  gradient: string;
  children: ReactNode;
}) {
  return (
    <motion.div
      key={gradient}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.7 }}
      className={`flex flex-col min-h-screen items-center justify-center ${gradient} p-8`}
      style={{ minHeight: "100vh" }}
    >
      {children}
    </motion.div>
  )
};