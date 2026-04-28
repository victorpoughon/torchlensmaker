import indicio

matches = indicio.search("H2O")
for shelf, book, page in matches:
    print(f"{shelf} | {book} | {page}")
