import { useCallback, useEffect, useState } from "react";
import { Mountain, Pause, Play, Trees } from "lucide-react";
import "./scenery.css";

type Scene = "lake" | "forest";
type SceneryPreferences = { scene: Scene; paused: boolean };

export type SceneryState = {
  scene: Scene;
  setScene: (scene: Scene) => void;
  paused: boolean;
  togglePaused: () => void;
  reducedMotion: boolean;
};

const STORAGE_KEY = "medguide-scenery";
const MOTION_QUERY = "(prefers-reduced-motion: reduce)";
const DEFAULT_PREFERENCES: SceneryPreferences = { scene: "lake", paused: false };

function readPreferences(): SceneryPreferences {
  if (typeof window === "undefined") return DEFAULT_PREFERENCES;
  try {
    const stored: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null");
    if (!stored || typeof stored !== "object" || Array.isArray(stored)) return DEFAULT_PREFERENCES;
    const value = stored as Record<string, unknown>;
    return {
      scene: value.scene === "forest" ? "forest" : "lake",
      paused: typeof value.paused === "boolean" ? value.paused : false,
    };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

function prefersReducedMotion() {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(MOTION_QUERY).matches
    : false;
}

export function useScenery(): SceneryState {
  const [preferences, setPreferences] = useState(readPreferences);
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(MOTION_QUERY);
    const onChange = () => setReducedMotion(query.matches);
    onChange();
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch {
      // The controls still work when browser storage is unavailable.
    }
  }, [preferences]);

  const setScene = useCallback((scene: Scene) => {
    setPreferences((current) => current.scene === scene ? current : { ...current, scene });
  }, []);

  const togglePaused = useCallback(() => {
    if (reducedMotion) return;
    setPreferences((current) => ({ ...current, paused: !current.paused }));
  }, [reducedMotion]);

  return {
    scene: preferences.scene,
    setScene,
    paused: preferences.paused || reducedMotion,
    togglePaused,
    reducedMotion,
  };
}

export function ScenicBackdrop({ scene, paused }: Pick<SceneryState, "scene" | "paused">) {
  return (
    <div className="scenic-backdrop" data-scene={scene} data-motion={paused ? "paused" : "playing"} aria-hidden="true">
      <div className="scenic-color-field scenic-color-base scenic-motion" />
      <div className="scenic-color-field scenic-color-cool scenic-motion" />
      <div className="scenic-color-field scenic-color-warm scenic-motion" />
      <div className="scenic-landscape" data-active={scene === "lake"}>
        <div className="scenic-landscape-image scenic-landscape-lake scenic-motion" />
      </div>
      <div className="scenic-landscape" data-active={scene === "forest"}>
        <div className="scenic-landscape-image scenic-landscape-forest scenic-motion" />
      </div>
      <div className="scenic-wash" />
      <div className="scenic-mist scenic-mist-near scenic-motion" />
      <div className="scenic-mist scenic-mist-far scenic-motion" />
      <div className="scenic-glow scenic-motion" />
      <span className="scenic-light scenic-light-one scenic-motion" />
      <span className="scenic-light scenic-light-two scenic-motion" />
      <span className="scenic-light scenic-light-three scenic-motion" />
    </div>
  );
}

export function SceneryControls({ scenery, attentionRequired = false }: { scenery: SceneryState; attentionRequired?: boolean }) {
  const motionLabel = attentionRequired ? "动态已暂停" : scenery.paused ? "开启动态" : "暂停动态";
  const pauseReason = attentionRequired ? "请优先按照紧急就医指引行动，背景动态已暂停" : scenery.reducedMotion ? "系统已启用减少动态，背景保持静止" : motionLabel;
  return (
    <div className="scenery-controls" role="group" aria-label="风景与动态设置">
      <div className="scenery-scene-switch" role="group" aria-label="选择风景">
        <button type="button" className="scenery-scene-button" aria-pressed={scenery.scene === "lake"} onClick={() => scenery.setScene("lake")}>
          <Mountain size={16} aria-hidden="true" />
          <span>山湖</span>
        </button>
        <button type="button" className="scenery-scene-button" aria-pressed={scenery.scene === "forest"} onClick={() => scenery.setScene("forest")}>
          <Trees size={16} aria-hidden="true" />
          <span>林间</span>
        </button>
      </div>
      <button
        type="button"
        className="scenery-motion-button"
        aria-label={attentionRequired || scenery.reducedMotion ? pauseReason : motionLabel}
        aria-pressed={!scenery.paused && !attentionRequired}
        title={pauseReason}
        disabled={scenery.reducedMotion || attentionRequired}
        onClick={scenery.togglePaused}
      >
        {scenery.paused && !attentionRequired ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}
        <span>{motionLabel}</span>
      </button>
    </div>
  );
}
