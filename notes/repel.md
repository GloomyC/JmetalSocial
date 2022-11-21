# Metody odpychania
Przemyślenia

Jakie cechy powinien mieć operator odpychania:
- nowe rozwiązanie musi być dalej od odpychającego przykładu
- potencjalnie: im bliżej do odpychającego rozwiązania tym silniejszy powninien być jego efekt

Spostrzeżenia:
- należy uważać na wielkość efektu, jeśli jest za duża, szybko rozwiązania zostaną wypchnięte na granice przedziału
- uwaga na konwersję typów (jmetal wykorzystuje wbudowane typy, konwersja z numpy kosztuje dużo)
- należy wykonać bardzo dużo powtórzeń dla każdego algorytmu, bo są to procesy stochastyczne, ze znaczącą dla porównania wariancją

Pomysły
- losowanie najgorszego rozwiązania z prawdopodobieństwem zależnym od odległości od aktulanego rozwiązania
- losowanie genów z prawdopodobieństwem zależnym od ich std (shared lub distinct)

TODO:
- uruchomienie wszystkiego na 4 powtórzeniach - done
