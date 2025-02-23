import React from "react";

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange }) => {
  // ボタン押下時に状態を反転するハンドラ
  const toggle = () => onChange(!checked);

  return (
    <button
      type="button"
      onClick={toggle}
      className={`relative inline-flex items-center h-8 w-16 rounded-full transition-colors duration-300 focus:outline-none 
        ${checked ? "bg-blue-300 nm-inset-blue-300" : "bg-gray-300 nm-inset-gray-300"}`}
    >
      {/* スクリーンリーダー向けのテキスト */}
      <span className="sr-only">Toggle Switch</span>
      <span
        className={`inline-block w-6 h-6 bg-white rounded-full shadow-md transform transition-transform duration-300 
          ${checked ? "translate-x-8" : "translate-x-1"}`}
      />
    </button>
  );
};

export default ToggleSwitch;
