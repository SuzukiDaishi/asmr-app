import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          200: "#edf2f7"
        }
      },
    },
  },
  plugins: [
    require("tw-neumorphism"),
  ],
  variants: {
    neumorphism: ['responsive'],
  }
} satisfies Config;
