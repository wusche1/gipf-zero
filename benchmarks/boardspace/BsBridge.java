// java -cp <classes> gipf.BsBridge <strategy> <seed>
// strategy is BoardSpace's robot level: 0 Dumbot, 1 Smartbot, 2 Bestbot, 3 (depth 3, never offered), 4 Weakbot.
// stdin:  "dump"
//         "push a1 b2"                    slide a single piece from an edge spot to its neighbour
//         "capture b2 e5 [e5 ...]"        remove the row on the line through these two spots; the listed GIPF pieces go too, the others stay
//         "done"                          confirm the end of the turn (only the player who inserted does this)
//         "go"                            the robot plays its whole turn the way GipfViewer runs it
//         "finish"                        play any legal moves until BoardSpace declares the game over
// stdout: one line per command, dumps separated by "|" (for "go": one after every move that changed the position, then the final one)
//         dump = "<state> <w|b to act> <37 spots b2..h5> <white reserve> <black reserve> <white lost> <black lost> <winner w|b|-> <robot ms> <moves the robot made>"
package gipf;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import lib.ExtendedHashtable;
import lib.Random;
import online.game.CommonMoveStack;
import online.game.commonMove;
import online.game.replayMode;
import online.game.export.ViewerProtocol;
import online.search.TreeViewerProtocol;

public class BsBridge implements GipfConstants {
	// commonRobot.InitRobot wants a live viewer and starts the robot thread; the search itself needs neither.
	static class Bot extends GipfPlay {
		long seed;
		public void InitRobot(ViewerProtocol v, ExtendedHashtable info, int strategy) { robotStrategy = strategy; rand = new Random(seed); }
		public TreeViewerProtocol getTreeViewer() { return null; }
	}

	static GipfBoard b;

	static GipfCell cell(String name) { return b.getCell(Character.toUpperCase(name.charAt(0)), Integer.parseInt(name.substring(1))); }
	static String name(GipfCell c) { return "" + Character.toLowerCase(c.col) + c.row; }
	static void exec(String move) { b.Execute(new Gipfmovespec(move, b.whoseTurn), replayMode.Replay); }

	static String position() {
		StringBuilder s = new StringBuilder();
		int[] reserve = { b.rack[0][0].height(), b.rack[1][0].height() };
		for (char col = 'A'; col <= 'I'; col++)
			for (int row = 1; row <= ZnInCol[col - 'A']; row++) {
				GipfCell c = b.getCell(col, row);
				int pl = c.height() == 0 ? -1 : c.topChip().color == GColor.W ? 0 : 1;
				if (c.isEdgeCell()) { if (pl >= 0) reserve[pl] += c.height(); }
				else s.append(pl < 0 ? '.' : (pl == 0 ? "oO" : "xX").charAt(c.height() - 1));
			}
		return s + " " + reserve[0] + " " + reserve[1] + " " + b.captures[1].height() + " " + b.captures[0].height();
	}

	static String dump(double ms, String did) {
		String winner = !b.GameOver() ? "-" : b.WinForPlayerNow(0) ? "w" : b.WinForPlayerNow(1) ? "b" : "-";
		return b.getState() + " " + "wb".charAt(b.whoseTurn) + " " + position() + " " + winner + " " + ms + " " + did;
	}

	static List<GipfCell> line(GipfCell a, GipfCell z, int dir) {
		GipfCell c = a;
		while (!c.isEdgeCell()) c = c.exitTo(dir + 3);
		List<GipfCell> cells = new ArrayList<>();
		for (cells.add(c), c = c.exitTo(dir); !c.isEdgeCell(); c = c.exitTo(dir)) cells.add(c);
		cells.add(c);
		return cells.contains(z) ? cells : null;
	}

	static void capture(String[] tok) {
		b.hasGipfCaptures(); // marks the rows of the player to act
		GipfCell a = cell(tok[1]), z = cell(tok[2]);
		Set<String> remove = new HashSet<>(Arrays.asList(tok).subList(3, tok.length));
		for (int dir = 0; dir < 3; dir++) {
			List<GipfCell> cells = line(a, z, dir);
			if (cells == null) continue;
			for (GipfCell c : cells)
				if (c.isGipf() && (c.rowcode & (1 << dir)) != 0 && c.preserved == remove.contains(name(c))) exec("Preserve " + c.col + " " + c.row);
			for (GipfCell c : cells)
				if (!c.isGipf() && c.rowcode == (1 << dir) && (c.isEdgeCell() || c.height() > 0)) { exec("Remove " + c.col + " " + c.row); return; }
		}
		throw new IllegalStateException("no row to capture there");
	}

	public static void main(String[] args) throws Exception {
		PrintStream out = System.out;
		System.setOut(System.err);
		b = new GipfBoard("gipf-standard", new int[] { 0, 1 }, GipfBoard.REVISION);
		exec("Start P0");
		Bot bot = new Bot();
		bot.seed = Long.parseLong(args[1]);
		bot.InitRobot(null, new ExtendedHashtable(), b, null, Integer.parseInt(args[0]));
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		for (String command; (command = in.readLine()) != null;) {
			String[] tok = command.trim().split(" +");
			List<String> dumps = new ArrayList<>();
			StringBuilder did = new StringBuilder();
			long started = System.nanoTime();
			try {
				if (tok[0].equals("push")) exec(("Slide " + tok[1].charAt(0) + " " + tok[1].substring(1) + " " + tok[2].charAt(0) + " " + tok[2].substring(1)).toUpperCase());
				else if (tok[0].equals("capture")) capture(tok);
				else if (tok[0].equals("done")) exec("Done");
				else if (tok[0].equals("finish"))
					for (int i = 0; i < 20 && !b.GameOver(); i++) {
						b.Execute(b.GetListOfMoves(new CommonMoveStack()).elementAt(0), replayMode.Replay);
						if (!b.GameOver() && b.mandatoryDoneState()) exec("Done");
					}
				else if (tok[0].equals("go")) {
					int who = b.whoseTurn;
					String last = position();
					for (int i = 0; i < 50 && !b.GameOver() && b.whoseTurn == who; i++) {
						bot.PrepareToMove(who);
						commonMove m = bot.DoFullMove();
						did.append(m.moveString().trim().replace(' ', '_')).append(',');
						b.Execute(m, replayMode.Replay);
						if (!position().equals(last)) dumps.add(dump(0, ""));
						last = position();
					}
				}
				dumps.add(dump((System.nanoTime() - started) / 1e6, did.toString()));
				out.println(String.join("|", dumps));
			} catch (Throwable e) {
				e.printStackTrace();
				out.println("ERR " + e + " after " + did + " at " + dump(0, ""));
			}
		}
		System.exit(0);
	}
}
