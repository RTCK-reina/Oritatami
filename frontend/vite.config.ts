import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: { '/api': 'http://127.0.0.1:47823' },
    },
    build: {
        outDir: 'dist',
        chunkSizeWarningLimit: 6000,
        sourcemap: false,
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                viewer: resolve(__dirname, 'viewer.html'),
            },
        },
    },
});
