// src/components/SentimentAnalyzer.jsx
import React, { useState } from 'react';
import { 
  Container,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Card,
  CardContent,
  Alert,
  Box
} from '@mui/material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)(({ theme, sentiment }) => ({
  marginTop: theme.spacing(3),
  borderLeft: `4px solid ${
    sentiment === 'positive' ? '#4caf50' :
    sentiment === 'negative' ? '#f44336' : '#ff9800'
  }`,
  transition: 'all 0.3s ease'
}));

const SentimentAnalyzer = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const MAX_LENGTH = 4000;

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      setResult(data.sentiment);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Sentiment Analysis Tool
      </Typography>

      <TextField
        fullWidth
        multiline
        rows={6}
        variant="outlined"
        label="Enter your text"
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, MAX_LENGTH))}
        helperText={`${text.length}/${MAX_LENGTH} characters`}
        sx={{ my: 2 }}
      />

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          startIcon={loading && <CircularProgress size={20} />}
        >
          {loading ? 'Analyzing...' : 'Analyze Text'}
        </Button>
      </Box>

      {result && (
        <StyledCard sentiment={result}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Analysis Result:
            </Typography>
            <Typography 
              variant="h4"
              color={
                result === 'positive' ? 'success.main' :
                result === 'negative' ? 'error.main' : 'warning.main'
              }
            >
              {result.charAt(0).toUpperCase() + result.slice(1)}
            </Typography>
          </CardContent>
        </StyledCard>
      )}
    </Container>
  );
};

export default SentimentAnalyzer;