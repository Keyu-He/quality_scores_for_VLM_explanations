// webpack.config.js
const path = require('path');
const webpack = require("webpack");

module.exports = (env, argv) => {
  const isDev = argv.mode === 'development';

  return {
    // multiple entry points, one for each “site”
    entry: {
      main: './src/main.ts',
      combined12: './src/main_combined12.ts',
      combined123: './src/main_combined123.ts',
    },

    devtool: isDev ? 'inline-source-map' : false,

    module: {
      rules: [{
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }]
    },

    resolve: {
      extensions: ['.tsx', '.ts', '.js']
    },

    output: {
      filename: '[name]/[name].js',        // e.g. combined12/combined12.js
      path: path.resolve(__dirname, 'web'),
      publicPath: '/',                      // so that <script src="/combined12/combined12.js"> works
    },

    plugins: [
      new webpack.DefinePlugin({
        _DEVMODE: JSON.stringify(isDev),
      }),
    ],

    performance: {
      hints: false,
    },

    // single devServer, one port
    devServer: {
      port: 8000,
      compress: true,

      // mount each folder at its sub-path
      static: [
        {
          directory: path.resolve(__dirname, 'web'),
          publicPath: '/',            // main build is at /
        },
        {
          directory: path.resolve(__dirname, 'web/combined12'),
          publicPath: '/combined12/',
        },
        {
          directory: path.resolve(__dirname, 'web/combined123'),
          publicPath: '/combined123/',
        },
      ],

      // client-side SPA fallback for each path
      historyApiFallback: {
        rewrites: [
          { from: /^\/$/,               to: '/index.html' },
          { from: /^\/combined12/,      to: '/combined12/index.html' },
          { from: /^\/combined123/,     to: '/combined123/index.html' },
        ],
      },

      open: true,                   // auto-open browser
      // openPage can specify which sub-path to open first:
      // openPage: ['combined12/'],
    },
  };
};
