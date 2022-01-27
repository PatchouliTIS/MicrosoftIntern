use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return;
    }
    let input_file_path = &args[1];
    let output_file_path = &args[2];

    // Open the file in read-only mode (ignoring errors).
    let input_file = File::open(input_file_path).unwrap();
    let reader = BufReader::new(input_file);

    let prefix = String::from("79b8984a784bf946918cd202b9374603");

    let mut output_file = std::fs::File::create(output_file_path).expect("create failed");

    // Read the file line by line using the lines() iterator from std::io::BufRead.
    for (index, line) in reader.lines().enumerate() {
        if let Ok(line) = line {
            let index = (1000000 + index).to_string();
            let trimmed_prefix = prefix.chars().take(32 - index.len()).collect::<String>();
            let new_line = format!("{}{}\t{}\n", trimmed_prefix, index, line);
            output_file.write(new_line.as_bytes()).expect("Error");
        }
    }
}
