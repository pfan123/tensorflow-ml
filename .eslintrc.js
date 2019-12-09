module.exports = {
    parser: '@typescript-eslint/parser',
    parserOptions: {
      ecmaFeatures: {
        jsx: true
      },
      project: './tsconfig.json'
    },
    plugins: ['@typescript-eslint'],
    extends: [
      'standard',
      'plugin:@typescript-eslint/eslint-recommended',
      'plugin:@typescript-eslint/recommended'
    ],
    rules: {
      '@typescript-eslint/triple-slash-reference': 0,
      '@typescript-eslint/no-var-requires': 0,
      '@typescript-eslint/no-use-before-define': 0,
      'arrow-body-style': 0,
    }
  }
