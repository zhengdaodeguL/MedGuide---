import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiProxyTarget = process.env.MEDGUIDE_API_PROXY_TARGET
    || process.env.MEDGUIDE_API_PROXY
    || env.MEDGUIDE_API_PROXY_TARGET
    || env.MEDGUIDE_API_PROXY
    || env.VITE_API_PROXY;

  return {
    plugins: [react()],
    server: {
      port: 5173,
      ...(apiProxyTarget ? {
        proxy: {
          "/api": {
            target: apiProxyTarget,
            changeOrigin: true,
          },
        },
      } : {}),
    },
  };
});
