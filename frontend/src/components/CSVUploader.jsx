// components/CSVUploader.jsx
import React, { useState } from 'react';
import { Button, CircularProgress, Box, Typography } from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
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

const CSVUploader = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    console.log('Selected file:', file); // Добавьте эту строку
  
    if (!file) {
      alert('Файл не выбран');
      return;
    }
  
    // Добавьте проверку размера файла
    if (file.size === 0) {
      alert('Файл пустой');
      return;
    }
  
    const formData = new FormData();
    formData.append('file', file);
  
    // Добавьте логирование FormData
    for (const [key, value] of formData.entries()) {
      console.log(key, value);
    }

    try {
      setIsProcessing(true);
      const response = await fetch('/api/analyze-csv', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Processing failed');
      
      // Скачивание результата
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'processed_results.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      
    } catch (error) {
      alert(error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Box sx={{ mt: 4, p: 3, border: '1px dashed grey' }}>
      <Typography variant="h6" gutterBottom>
        Batch Processing (CSV)
      </Typography>
      
      <Button
      component="label"
      variant="outlined"
      color="secondary"
      startIcon={<CloudUpload />}
      disabled={isProcessing}
      sx={{ width: { xs: '100%', sm: 'auto' }, py: 1.5 }}
    >
      {isProcessing ? (
        <>
          <CircularProgress size={24} sx={{ mr: 1 }} />
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
      
      <Typography variant="body2" sx={{ mt: 1 }}>
        CSV format: text_column (max 1000 rows, 5MB)
      </Typography>
    </Box>
  );
};

export default CSVUploader;