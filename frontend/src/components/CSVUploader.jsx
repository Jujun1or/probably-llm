import React, { useState, useEffect } from 'react';
import { Button, CircularProgress, Box, Typography, LinearProgress, Divider } from '@mui/material';
import { CloudUpload, Download } from '@mui/icons-material';
import { styled } from '@mui/system';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

// Стили для анимации заполнения баров
const ProgressBarWithAnimation = styled(LinearProgress)(({ theme }) => ({
  height: 8,
  borderRadius: 5,
  backgroundColor: '#333',
  '& .MuiLinearProgress-bar': {
    borderRadius: 5,
    transition: 'transform 1s ease-out', // Анимация заполнения
  },
}));

const CSVUploader = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState({ positive: 0, neutral: 0, negative: 0, total: 0 });
  const [animatedValues, setAnimatedValues] = useState({ positive: 0, neutral: 0, negative: 0 });

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    console.log('Selected file:', file);

    if (!file) {
      alert('Файл не выбран');
      return;
    }

    if (file.size === 0) {
      alert('Файл пустой');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsProcessing(true);
      setProgress(0);

      const response = await fetch('/api/analyze-csv', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Processing failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
      setProgress(100);

      // Чтение CSV-файла для подсчета результатов
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const rows = text.split('\n').slice(1); // Пропускаем заголовок
        let positive = 0, neutral = 0, negative = 0;

        rows.forEach(row => {
          const sentiment = row.split(',')[1]; // Предположим, что sentiment находится во второй колонке
          if (sentiment === 'positive') positive++;
          else if (sentiment === 'neutral') neutral++;
          else if (sentiment === 'negative') negative++;
        });

        setResults({ positive, neutral, negative, total: rows.length });
      };
      reader.readAsText(file);
      
    } catch (error) {
      alert(error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Анимация заполнения баров
  useEffect(() => {
    if (showResults) {
      const positiveValue = (results.positive / results.total) * 100;
      const neutralValue = (results.neutral / results.total) * 100;
      const negativeValue = (results.negative / results.total) * 100;

      // Запускаем анимацию для каждого бара с задержкой
      setTimeout(() => {
        setAnimatedValues({ positive: positiveValue, neutral: 0, negative: 0 });
      }, 100);

      setTimeout(() => {
        setAnimatedValues((prev) => ({ ...prev, neutral: neutralValue }));
      }, 1200);

      setTimeout(() => {
        setAnimatedValues((prev) => ({ ...prev, negative: negativeValue }));
      }, 2400);
    } else {
      setAnimatedValues({ positive: 0, neutral: 0, negative: 0 });
    }
  }, [showResults, results]);

  return (
    <Box sx={{ mt: 4, p: 3, border: '1px dashed #555', borderRadius: 2, backgroundColor: '#1e1e1e' }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#fff' }}>
        Batch Processing (CSV)
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            component="label"
            variant="outlined"
            color="secondary"
            startIcon={<CloudUpload />}
            disabled={isProcessing}
            sx={{ width: { xs: '100%', sm: 'auto' }, py: 1.5, color: '#fff', borderColor: '#555' }}
          >
            {isProcessing ? (
              <>
                <CircularProgress size={24} sx={{ mr: 1, color: '#fff' }} />
                Processing...
              </>
            ) : (
              'Upload CSV'
            )}
            <VisuallyHiddenInput 
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
            />
          </Button>

          {isProcessing && (
            <Box sx={{ width: '30%' }}>
              <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 5 }} />
            </Box>
          )}
        </Box>

        {downloadUrl && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 2, borderColor: '#555' }} />
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={() => setShowResults(!showResults)}
                sx={{ backgroundColor: '#ff9800', '&:hover': { backgroundColor: '#f57c00' } }}
              >
                {showResults ? 'Hide Results' : 'Show Results'}
              </Button>

              {showResults && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    Positive: {results.positive} / {results.total}
                  </Typography>
                  <ProgressBarWithAnimation 
                    variant="determinate" 
                    value={animatedValues.positive} 
                    sx={{ '& .MuiLinearProgress-bar': { backgroundColor: '#4caf50' } }}
                  />

                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    Neutral: {results.neutral} / {results.total}
                  </Typography>
                  <ProgressBarWithAnimation 
                    variant="determinate" 
                    value={animatedValues.neutral} 
                    sx={{ '& .MuiLinearProgress-bar': { backgroundColor: '#ff9800' } }}
                  />

                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    Negative: {results.negative} / {results.total}
                  </Typography>
                  <ProgressBarWithAnimation 
                    variant="determinate" 
                    value={animatedValues.negative} 
                    sx={{ '& .MuiLinearProgress-bar': { backgroundColor: '#f44336' } }}
                  />
                </Box>
              )}
            </Box>
          </>
        )}
      </Box>

      {downloadUrl && (
        <Button
          variant="contained"
          color="primary"
          startIcon={<Download />}
          sx={{ mt: 2, backgroundColor: '#4caf50', '&:hover': { backgroundColor: '#388e3c' } }}
          onClick={() => {
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = 'processed_results.csv';
            document.body.appendChild(a);
            a.click();
            a.remove();
          }}
        >
          Download Results
        </Button>
      )}
      
      <Typography variant="body2" sx={{ mt: 1, color: '#aaa' }}>
        CSV format: text_column (max 1000 rows, 5MB)
      </Typography>
    </Box>
  );
};

export default CSVUploader;