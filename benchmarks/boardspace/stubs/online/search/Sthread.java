package online.search;
import lib.Task;
import online.game.commonMove;
public class Sthread {
	public long evalTime;
	public Sthread(Search_Driver driver, RobotProtocol robot, int index, ThreadPoolMonitor monitor) { throw new UnsupportedOperationException("stub: single-threaded search only"); }
	public static void waitForIdle(Sthread[] pool) {}
	public static boolean waitIfStuck(ThreadPoolMonitor monitor) { return false; }
	public void start() {}
	public void makeMove(commonMove m) {}
	public void unmakeMove(commonMove m) {}
	public void setExit() {}
	public void stopEvaling() {}
	public void setPaused(boolean v) {}
	public boolean startEval(commonMove m) { return false; }
	public void cancelEval() {}
	public commonMove finishEval() throws Throwable { return null; }
	public RobotProtocol getRobot() { return null; }
	public void doSomething(Task t) {}
}
