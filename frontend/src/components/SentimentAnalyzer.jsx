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
  Grid,
  Stack
} from '@mui/material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)(({ theme, sentiment }) => ({
  marginTop: theme.spacing(2),
  borderLeft: `4px solid ${
    sentiment === 'Positive' ? '#4caf50' :
    sentiment === 'Negative' ? '#f44336' : '#ff9800'
  }`,
  transition: 'all 0.3s ease',
  backgroundColor: '#1e1e1e',
  width: '100%', // Ширина блока будет адаптироваться
  height: '120px', // Высота блока
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
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
      <Typography variant="h4" gutterBottom align="center" sx={{ color: '#fff' }}>
        Sentiment Analysis Tool
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={6}
            variant="outlined"
            label="Enter your text"
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_LENGTH))}
            helperText={`${text.length}/${MAX_LENGTH} characters`}
            sx={{ my: 2, backgroundColor: '#1e1e1e', color: '#fff' }}
            InputProps={{
              style: { color: '#fff' },
            }}
            InputLabelProps={{
              style: { color: '#aaa' },
            }}
          />
        </Grid>

        <Grid item xs={12}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3} alignItems="flex-start">
            <Box sx={{ flex: 1, width: '100%' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
                sx={{ 
                  width: { xs: '100%', sm: '200px' }, // Адаптивная ширина
                  py: 1.5, 
                  backgroundColor: '#4caf50',
                  '&:hover': { backgroundColor: '#388e3c' },
                  transition: 'all 0.3s ease',
                }}
              >
                {loading ? <CircularProgress size={24} /> : 'Analyze Text'}
              </Button>
            </Box>

            {result && (
              <Box sx={{ width: { xs: '100%', sm: '400px' } }}> {/* Адаптивная ширина */}
                <StyledCard sentiment={result}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ color: '#fff', textAlign: 'center' }}>
                      Analysis Result:
                    </Typography>
                    <Typography 
                      variant="h4"
                      color={
                        result === 'Positive' ? 'success.main' :
                        result === 'Negative' ? 'error.main' : 'warning.main'
                      }
                      sx={{ textAlign: 'center' }}
                    >
                      {result.charAt(0).toUpperCase() + result.slice(1)}
                    </Typography>
                  </CardContent>
                </StyledCard>
              </Box>
            )}
          </Stack>
        </Grid>

        <Grid item xs={12}>
          <CSVUploader />
        </Grid>
      </Grid>
    </Container>
  );
};

export default SentimentAnalyzer;