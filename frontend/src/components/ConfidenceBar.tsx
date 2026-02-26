import { motion } from "framer-motion";

interface ConfidenceBarProps {
  label: string;
  value: number;
  index: number;
}

const ConfidenceBar = ({ label, value, index }: ConfidenceBarProps) => {
  const percentage = Math.round(value * 100);

  const getBarColor = () => {
    if (percentage >= 70) return "bg-destructive";
    if (percentage >= 50) return "bg-warning";
    return "bg-success";
  };

  const getRiskLabel = () => {
    if (percentage >= 70) return "High";
    if (percentage >= 50) return "Moderate";
    return "Low";
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-foreground">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{getRiskLabel()}</span>
          <span className="font-mono text-foreground">{percentage}%</span>
        </div>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${getBarColor()}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, delay: index * 0.15, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};

export default ConfidenceBar;
