import { useState } from "react";
import HeroSection from "@/sections/HeroSection";
import UploadSection from "@/sections/UploadSection";
import ResultsSection from "@/sections/ResultsSection";
import ChatSection from "@/sections/ChatSection";
import type { AnalysisResult } from "@/sections/UploadSection";

const Index = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);

  return (
    <div className="min-h-screen bg-background">
      <HeroSection />
      <UploadSection onResult={setResult} />
      <ResultsSection result={result} />
      <ChatSection />

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-border">
        <p className="text-center text-xs text-muted-foreground">
          MedXScan AI — Clinical-grade radiology analysis. For research and educational purposes only.
        </p>
      </footer>
    </div>
  );
};

export default Index;
