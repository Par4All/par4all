
typedef int points ;
 
void main() {
  points high_score;
  points your_score;
  
  high_score = 100;
  your_score = 110;
  
  if (your_score > high_score)
    high_score = your_score;
}
