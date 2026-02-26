import { motion } from "framer-motion";
import { Activity, Shield, Brain } from "lucide-react";

const HeroSection = () => {
  return (
    <section className="min-h-screen flex flex-col items-center justify-center px-6 relative overflow-hidden">
      {/* Subtle grid background */}
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: "linear-gradient(hsl(217 91% 60%) 1px, transparent 1px), linear-gradient(90deg, hsl(217 91% 60%) 1px, transparent 1px)",
        backgroundSize: "60px 60px"
      }} />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-3xl relative z-10"
      >
        <div className="flex items-center justify-center gap-2 mb-6">
          <Activity className="w-6 h-6 text-primary" />
          <span className="text-sm font-medium tracking-widest uppercase text-muted-foreground">
            AI-Powered Radiology
          </span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
          <span className="gradient-text">MedXScan</span>{" "}
          <span className="text-foreground">AI</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          Clinical-grade chest X-ray analysis powered by DenseNet-121 deep learning.
          Detect lung diseases with Grad-CAM visual explanations.
        </p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="flex flex-wrap items-center justify-center gap-6 mt-12"
        >
          {[
            { icon: Brain, label: "DenseNet-121" },
            { icon: Shield, label: "Clinical Grade" },
            { icon: Activity, label: "Real-time Analysis" },
          ].map(({ icon: Icon, label }) => (
            <div
              key={label}
              className="flex items-center gap-2 text-sm text-muted-foreground"
            >
              <Icon className="w-4 h-4 text-primary" />
              <span>{label}</span>
            </div>
          ))}
        </motion.div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-10"
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="w-5 h-8 rounded-full border-2 border-muted-foreground/30 flex justify-center pt-1.5">
          <div className="w-1 h-2 rounded-full bg-muted-foreground/50" />
        </div>
      </motion.div>
    </section>
  );
};

export default HeroSection;
