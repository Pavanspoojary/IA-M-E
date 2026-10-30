const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

// Test that public/browser_extractor.js exists and is valid syntax
const code = fs.readFileSync('public/browser_extractor.js', 'utf8');
console.log('browser_extractor.js length:', code.length, 'bytes');

// Check for syntax errors by parsing with Function constructor
try {
  new Function('window', 'document', 'ort', 'cv', 'XLSX', code);
  console.log('Syntax check passed!');
} catch (err) {
  console.error('Syntax error:', err);
}
