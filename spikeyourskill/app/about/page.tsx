'use client';

import TransitionBackground from "../components/TransitionBackground";
import { useState } from "react";
import Image from "next/image";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
const images = ['/af.png', '/ah.png',  '/gor.png'];

export default function Page() {
  const [current, setCurrent] = useState(1);
  const [isFocus, setFocus] = useState(false);
  const onPrevClick = () => {
    if (current > 0) {
      setCurrent(current - 1);
    }
  };

  const onNextClick = () => {
    if (current < images.length - 1) {
      setCurrent(current + 1);
    }
  };
  return (
    <TransitionBackground gradient="bg-gradient-to-br from-[#2d014d] to-black transition-colors duration-700">
      <title>Spike Your Skill</title>
      <h1 className="text-white text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 text-center">About Spike Your Skill</h1>
      <div className="relative w-full max-w-3xl aspect-video mx-auto mb-6">
        <video controls
        className="w-full h-full rounded-2xl border-4xl object-cover"
        >
          <source src="/demo.mp4" type="video/mp4"/>
          <track

          />
          Your browser does not support the video tag.
        </video>
      </div>
      <p className="text-white text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold mb-4 text-center">App Photos</p>
      
      <MotionConfig transition = {{duration: 0.7, ease:[0.32, 0.72, 0, 1]}}>
      <div className = "relative w-full max-w-3xl mx-auto overflow-hidden">
        <AnimatePresence>
        {isFocus && (
        <motion.div className = "absolute left-2 right-2 top-1/2 -translate-y-1/2 flex justify-between z-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onHoverStart={() => setFocus(true)}
          onHoverEnd={() => setFocus(false)}
        >
          {
            current > 0 ? (
          <button onClick={onPrevClick}>
              <Image src="/left.png" alt="Left" width={32} height={32} />
          </button>
          ) : (
            <div style={{width:32, height:32}}/>
          )}
          { current < images.length - 1 ? (
          <button onClick={onNextClick}>
              <Image src="/right.png" alt="Right" width={32} height={32} />
          </button>
          ) : (
            <div style = {{width:32, height:32}}/>
          )}   
        </motion.div>
        )}
        </AnimatePresence>
        <motion.div className = 'flex gap-4 flex-nowrap'
          animate={{ x: `calc(-${current * 100}% - ${current}rem)` }}
          onHoverStart={() => setFocus(true)}
          onHoverEnd={() => setFocus(false)}
        >
          {[...images].map((image, idx) => (
            <img key = {idx} src={image} alt = "image not loaded" className = "object-cover aspect-[16/9]"></img>
          ))}
        </motion.div>
      </div>
      </MotionConfig>
      {/* About page content here */}
    </TransitionBackground>
  );
}