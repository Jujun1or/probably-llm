// src/components/SentimentAnalyzer.jsx
import React, { useState } from 'react';
import CSVUploader from './CSVUploader';
import { 
  Container,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Card,
  CardContent,
  Alert,
  Box,
  Stack,
  Divider
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
      <Typography variant="h4" gutterBottom align="center">
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
        sx={{ my: 3 }}
      />

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Stack 
        direction={{ xs: 'column', sm: 'row' }}
        spacing={3}
        justifyContent="center"
        alignItems="center"
        sx={{ mb: 4 }}
      >
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          sx={{ width: { xs: '100%', sm: 'auto' }, py: 1.5 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Analyze Text'}
        </Button>

        <Divider orientation="vertical" flexItem sx={{ mx: 2, display: { xs: 'none', sm: 'block' } }} />

        <CSVUploader />
      </Stack>


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