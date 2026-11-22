import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";

// Copia la carpeta static/ (Three.js, HDRI, modelos GLTF) dentro de dist/static/ al compilar
function copyStaticPlugin() {
  return {
    name: "copy-static-assets",
    closeBundle() {
      const srcDir = path.resolve("static");
      const destDir = path.resolve("dist/static");
      if (fs.existsSync(srcDir)) {
        fs.cpSync(srcDir, destDir, { recursive: true });
      }
    }
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), copyStaticPlugin()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/static": "http://localhost:8000"
    }
  }
});
