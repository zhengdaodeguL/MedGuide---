import { useState } from "react";
import { ArrowUpRight, ClipboardList, HeartPulse, Pill, Stethoscope } from "lucide-react";

export const QUESTION_STARTER_CATEGORIES = [
  {
    id: "symptoms",
    label: "身体不适",
    icon: HeartPulse,
    example: "最近____有些不舒服，描述的时候从哪里说起？",
    prompts: [
      {
        title: "不舒服，从哪里说起",
        description: "先说部位和时间，再慢慢补充细节。",
        prompt: "我想整理最近的不适：不舒服的部位是____，从____开始，出现频率是____。还需要补充哪些信息？",
      },
      {
        title: "反复出现，想理清规律",
        description: "把零散的感受，记成有用的线索。",
        prompt: "我想记录____出现的规律：一般在____时发生，每次大约____，最近的变化是____。有哪些信息值得一起记录？",
      },
    ],
  },
  {
    id: "medication",
    label: "用药疑问",
    icon: Pill,
    example: "关于____这个药，使用前有哪些事情需要确认？",
    prompts: [
      {
        title: "拿到药，先了解什么",
        description: "看懂常见用途，也弄清注意事项。",
        prompt: "我想了解____（药品名称）的常见用途和注意事项。相关情况是____。哪些问题应该向医生或药师确认？",
      },
      {
        title: "几种药，想一起核对",
        description: "准备一份清楚的用药咨询清单。",
        prompt: "我想向医生或药师核对药品或保健品：涉及的名称是____，使用情况是____。咨询相互作用时，还需要提供哪些信息？",
      },
    ],
  },
  {
    id: "examination",
    label: "检查解读",
    icon: ClipboardList,
    example: "报告上的____是什么意思，解读时还要看哪些信息？",
    prompts: [
      {
        title: "报告里，有个词看不懂",
        description: "带上项目、结果和参考范围来聊。",
        prompt: "检查报告中有一个项目叫____，结果是____，参考范围是____。请解释这个项目的一般含义，以及解读时还需要哪些信息。",
      },
      {
        title: "检查前，想准备充分",
        description: "把机构要求和自己的疑问列清楚。",
        prompt: "我想了解____检查的常见注意事项。检查机构给出的要求是____。有哪些准备事项需要向该机构再次确认？",
      },
    ],
  },
  {
    id: "visit",
    label: "就医准备",
    icon: Stethoscope,
    example: "准备为____去一趟医院，有哪些资料和问题可以提前整理？",
    prompts: [
      {
        title: "去医院前，带些什么",
        description: "就诊方向、现有资料，一起理一理。",
        prompt: "我准备因____前往医院，希望了解可能的就诊方向。已有的相关检查资料是____（没有可填“无”）。还需要准备哪些信息或材料？",
      },
      {
        title: "看诊时，先问哪几件事",
        description: "把最在意的问题，提前写下来。",
        prompt: "我想把就诊时要问的问题提前理清：最想确认的是____，目前掌握的信息是____。请帮我整理一份简短的提问清单。",
      },
    ],
  },
] as const;

type QuestionStartersProps = {
  compact?: boolean;
  disabled?: boolean;
  onChoose?: (prompt: string) => void;
};

export function QuestionStarters({ compact = false, disabled = false, onChoose }: QuestionStartersProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const selected = QUESTION_STARTER_CATEGORIES[activeIndex];
  const SelectedIcon = selected.icon;

  return (
    <section className={`question-starters${compact ? " compact" : ""}`} aria-label="从一个问题开始">
      <div className="starter-tabs" role="group" aria-label="选择提问主题">
        {QUESTION_STARTER_CATEGORIES.map((category, index) => {
          const CategoryIcon = category.icon;
          return (
            <button
              key={category.id}
              type="button"
              className={`starter-tab${index === activeIndex ? " active" : ""}`}
              aria-pressed={index === activeIndex}
              disabled={disabled}
              onClick={() => setActiveIndex(index)}
            >
              <CategoryIcon size={15} aria-hidden="true" />
              <span>{category.label}</span>
            </button>
          );
        })}
      </div>

      <div className="starter-prompts" aria-label={`${selected.label}的提问示例`}>
        {compact ? (
          <div className="starter-prompt">
            <span className="starter-prompt-icon"><SelectedIcon size={19} aria-hidden="true" /></span>
            <div className="starter-prompt-copy">
              <strong>比如，你可以这样问</strong>
              <span>{selected.example}</span>
            </div>
          </div>
        ) : selected.prompts.map((item) => (
          <button
            key={item.title}
            type="button"
            className="starter-prompt"
            disabled={disabled || !onChoose}
            aria-label={`将“${item.title}”填入输入框`}
            onClick={() => onChoose?.(item.prompt)}
          >
            <span className="starter-prompt-icon"><SelectedIcon size={19} aria-hidden="true" /></span>
            <span className="starter-prompt-copy">
              <strong>{item.title}</strong>
              <span>{item.description}</span>
            </span>
            <span className="starter-prompt-action"><ArrowUpRight size={17} aria-hidden="true" /></span>
          </button>
        ))}
      </div>

      <p className="starter-caption">{compact ? "登录后开始整理" : "点选问题，补充你的情况，再发送。"}</p>
    </section>
  );
}
