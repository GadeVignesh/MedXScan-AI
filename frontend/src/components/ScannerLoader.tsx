import { motion } from "framer-motion";

const ScannerLoader = () => {
  return (
    <div className="flex flex-col items-center gap-6 py-12">
      <div className="relative w-32 h-32 rounded-lg border border-border overflow-hidden bg-secondary/50">
        <motion.div
          className="absolute left-0 w-full h-1 scanner-line"
          animate={{ top: ["0%", "100%", "0%"] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="w-12 h-12 rounded-full border-2 border-primary border-t-transparent"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        </div>
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">Analyzing X-Ray</p>
        <p className="text-xs text-muted-foreground mt-1">Running DenseNet-121 inference...</p>
      </div>
    </div>
  );
};

export default ScannerLoader;
