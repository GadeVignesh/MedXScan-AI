import { motion } from "framer-motion";
import ChatBox from "@/components/ChatBox";

const ChatSection = () => {
  return (
    <section className="py-24 px-6">
      <div className="max-w-2xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-bold text-foreground mb-2 text-center">
            AI Medical Assistant
          </h2>
          <p className="text-sm text-muted-foreground mb-8 text-center">
            Ask questions about radiology, lung conditions, or your results
          </p>

          <ChatBox />
        </motion.div>
      </div>
    </section>
  );
};

export default ChatSection;
