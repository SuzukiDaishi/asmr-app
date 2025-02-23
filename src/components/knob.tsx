import React, { useState, MouseEvent, ChangeEvent } from "react";

interface KnobProps {
  mx?: number;
  my?: number;
  minValue?: number;
  maxValue?: number;
  value: number;
  onChange: (newValue: number) => void;
  size?: number;
  scale?: "linear" | "log";
  logSensitivity?: number;
}

const Knob: React.FC<KnobProps> = ({
  mx = 200,
  my = 200,
  minValue = 0,
  maxValue = 100,
  value,
  onChange,
  size = 120,
  scale = "linear",
  logSensitivity = 0.01,
}) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [startX, setStartX] = useState<number>(0);
  const [startY, setStartY] = useState<number>(0);

  // --- 対数変換（必要に応じてそのまま）---
  const toLog = (v: number): number =>
    v >= 0 ? Math.log(v + 1) : -Math.log(-v + 1);
  const fromLog = (v: number): number =>
    v >= 0 ? Math.exp(v) - 1 : -(Math.exp(-v) - 1);

  const handleMouseDown = (event: MouseEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest("svg")) {
      setIsDragging(true);
      setStartX(event.clientX);
      setStartY(event.clientY);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseMove = (event: MouseEvent<HTMLDivElement>) => {
    if (!isDragging || event.buttons !== 1) return;

    const x = event.clientX;
    const y = event.clientY;
    const dx = x - startX;      // 右方向が正
    const dy = startY - y;      // 上方向が正

    // どちらのドラッグ量が大きいかで決定（水平 or 垂直）
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    const df =
      absDx > absDy
        ? (dx / mx) * (maxValue - minValue)
        : (dy / my) * (maxValue - minValue);

    let newValue: number;
    if (scale === "log") {
      const currentLog = toLog(value);
      const newLog = currentLog + df * logSensitivity;
      newValue = fromLog(newLog);
    } else {
      newValue = value + df;
    }

    newValue = Math.max(minValue, Math.min(maxValue, newValue));
    onChange(newValue);

    setStartX(x);
    setStartY(y);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    let inputStr = event.target.value;
    inputStr = inputStr.match(/^-?\d*\.?\d*/)?.[0] || "";
    const newValue = Number(inputStr);
    onChange(newValue);
  };

  // ゲージ表示用の割合 [0.0 ~ 1.0]
  const rotationPercentage = (value - minValue) / (maxValue - minValue);

  // テキスト表示用フォーマット
  const formatValue = (num: number): string => {
    const str = num.toString();
    return str.length > 6 ? str.slice(0, 6) : str;
  };

  return (
    <div
      className="flex flex-col justify-center items-center"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <div
        className="knob-container relative rounded-full bg-gray-300 shadow-xl flex justify-center items-center"
        style={{ width: `${size}px`, height: `${size}px` }}
      >
        <svg className="absolute w-full h-full" viewBox="0 0 100 100">
          {/* 
            Base track: 下地の円。色を分かりやすく "gray" にする例。
          */}
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="#ccc"
            strokeWidth="10"
            fill="none"
          />
          {/*
            Progress track:
            - 一番下(6時位置)を最小値にしたいので rotate(90 50 50) で円を回転し
              "右回り" で値が大きくなる見た目に
            - strokeDashoffset = 282.6 * (1 - rotationPercentage) として
              値が増えるほど offset が小さくなり、ゲージが長くなる
          */}
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="#007bff"
            strokeWidth="10"
            fill="none"
            strokeDasharray="282.6"
            strokeDashoffset={282.6 * (1 - rotationPercentage)}
            transform="rotate(90 50 50)"
          />
        </svg>
      </div>
      <input
        type="number"
        value={formatValue(value)}
        onChange={handleInputChange}
        className="mt-4 w-20 h-10 text-center text-xl border-none rounded-lg
                   nm-flat-gray-200
                   focus:nm-inset-gray-200
                   focus:outline-none
                   transition duration-200 ease-in-out"
      />
    </div>
  );
};

export default Knob;
